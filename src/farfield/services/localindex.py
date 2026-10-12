"""Literature search against a local copy of arXiv's metadata: no key, no quota, no rate limit.

The index is built by ``scripts/literature/harvest_arxiv.py`` (arXiv's OAI-PMH bulk channel)
into SQLite with an FTS5 table over title and abstract. Search ranks by BM25, title weighted
three times the abstract, and can stop at a publication date — which a benchmark with an
idea-conception date needs, and which a live API cannot be trusted to apply the same way twice.

What it does not have: citation links. ``related`` stands in for "the references of a paper"
with the papers most similar to it that appeared before it; callers that report results say so.

``LocalFeed`` gives ``farfield.scientist`` the same interface as ``livefeed.CompositeFeed``.
Its rows are arXiv's own records, so they skip the online existence check the live feed runs.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .livefeed import FreshWork

_STOP = set("""a an and are as at be by can do does for from has have how in into is it its of on or
that the their this to under using via was we what when where which while with within without
not no than then these those our new based study approach method methods paper results show""".split())
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")


def index_path() -> Path | None:
    chosen = os.environ.get("FARFIELD_LITERATURE_INDEX", "").strip()
    return Path(chosen) if chosen and Path(chosen).is_file() else None


def terms(text: str, limit: int = 12) -> list[str]:
    seen: list[str] = []
    for word in _WORD.findall(text or ""):
        low = word.lower().strip("-")
        if len(low) > 2 and low not in _STOP and low not in seen:
            seen.append(low)
    return seen[:limit]


def _match(words: Iterable[str]) -> str:
    return " OR ".join('"' + word.replace('"', "") + '"' for word in words)


class LocalIndex:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._local = threading.local()

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            db.row_factory = sqlite3.Row
            self._local.db = db
        return db

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "title": row["title"], "abstract": row["abstract"],
                "categories": row["categories"], "published": row["created"], "doi": row["doi"],
                "authors": row["authors"]}

    def search(self, query: str, *, limit: int = 10, before: str = "", after: str = "",
               exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
        words = terms(query)
        if not words:
            return []
        clauses, params = ["papers_fts MATCH ?"], [_match(words)]
        if before:
            clauses.append("p.created <= ?")
            params.append(before)
        if after:
            clauses.append("p.created >= ?")
            params.append(after)
        skip = list(exclude)
        if skip:
            clauses.append(f"p.id NOT IN ({','.join('?' * len(skip))})")
            params += skip
        sql = (f"SELECT p.* FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid "
               f"WHERE {' AND '.join(clauses)} ORDER BY bm25(papers_fts, 3.0, 1.0) LIMIT ?")
        return [self._row(row) for row in self._db().execute(sql, (*params, int(limit)))]

    def get(self, paper_id: str) -> dict[str, Any] | None:
        row = self._db().execute("SELECT * FROM papers WHERE id = ?", (normalise_id(paper_id),)).fetchone()
        return self._row(row) if row else None

    def related(self, paper_id: str, *, limit: int = 15, before: str = "") -> list[dict[str, Any]]:
        """The papers most similar to ``paper_id`` that appeared before it: a stand-in for its
        reference list, which the arXiv metadata does not carry."""
        paper = self.get(paper_id)
        if paper is None:
            return []
        cutoff = min(filter(None, (paper["published"], before))) if before else paper["published"]
        return self.search(f"{paper['title']} {paper['abstract'][:400]}", limit=limit, before=cutoff,
                           exclude=[paper["id"]])

    def count(self) -> int:
        return int(self._db().execute("SELECT count(*) FROM papers").fetchone()[0])


class RemoteIndex:
    """The same three calls, answered by ``scripts/literature/serve_index.py`` on this machine,
    which reranks BM25's candidates with a neural model."""

    def __init__(self, url: str, mode: str = "") -> None:
        self.url, self.mode = url.rstrip("/"), mode

    def _get(self, path: str, **params: Any) -> Any:
        params = {k: v for k, v in params.items() if v not in ("", None)}
        if self.mode:
            params["mode"] = self.mode
        with urllib.request.urlopen(f"{self.url}{path}?{urllib.parse.urlencode(params)}", timeout=120) as resp:
            return json.load(resp)

    def search(self, query: str, *, limit: int = 10, before: str = "", after: str = "",
               exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
        skip = set(exclude)
        rows = self._get("/search", q=query, limit=int(limit) + len(skip), before=before, after=after)
        return [row for row in rows if row["id"] not in skip][: int(limit)]

    def get(self, paper_id: str) -> dict[str, Any] | None:
        return self._get("/paper", id=normalise_id(paper_id))

    def related(self, paper_id: str, *, limit: int = 15, before: str = "") -> list[dict[str, Any]]:
        return self._get("/related", id=normalise_id(paper_id), limit=limit, before=before)

    def count(self) -> int:
        return int(self._get("/health")["papers"])


def open_index(where: str = "") -> "LocalIndex | RemoteIndex | None":
    """The literature index to use: a server URL, a SQLite file, or — with no argument — whatever
    FARFIELD_LITERATURE_SERVER or FARFIELD_LITERATURE_INDEX names."""
    where = where or os.environ.get("FARFIELD_LITERATURE_SERVER", "").strip()
    if where.startswith(("http://", "https://")):
        return RemoteIndex(where, os.environ.get("FARFIELD_LITERATURE_MODE", "").strip())
    path = Path(where) if where else index_path()
    return LocalIndex(path) if path and Path(path).is_file() else None


def normalise_id(paper_id: str) -> str:
    text = str(paper_id or "").strip()
    for prefix in ("arXiv:", "arxiv:", "https://arxiv.org/abs/", "http://arxiv.org/abs/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return re.sub(r"v\d+$", "", text)


class LocalFeed:
    """``survey_around`` over the local index, for ``farfield.scientist``'s literature step."""

    verify_literature = False      # rows are arXiv's own metadata, not model- or web-supplied

    def __init__(self, index: "LocalIndex | RemoteIndex", *, before: str = "") -> None:
        self.index = index
        self.before = before or os.environ.get("FARFIELD_LITERATURE_CUTOFF", "").strip()
        self.last_sources_blocked: list[dict[str, Any]] = []

    def survey_around(self, concept_a: str, concept_b: str, *, per_side: int = 4, claim: str = "",
                      topic: str = "") -> list[FreshWork]:
        seen: dict[str, FreshWork] = {}
        for query in (topic or concept_a, f"{concept_a} {concept_b}", claim):
            if not str(query or "").strip():
                continue
            for row in self.index.search(query, limit=per_side * 2, before=self.before):
                seen.setdefault(row["id"], FreshWork(
                    title=row["title"], published=row["published"], arxiv_id=row["id"],
                    abstract=row["abstract"], source="arxiv-local", url=f"https://arxiv.org/abs/{row['id']}"))
        return list(seen.values())[: per_side * 3]
