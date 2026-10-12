"""Build a local, searchable copy of arXiv's metadata, so literature search has no quota.

    python3.11 scripts/literature/harvest_arxiv.py --db /data/literature/arxiv.sqlite --sets cs stat

Harvests through arXiv's OAI-PMH interface (https://oaipmh.arxiv.org/oai), the channel arXiv
provides for bulk metadata: identifier, title, abstract, categories, first-version date, DOI.
Each set is harvested in pages linked by resumption tokens; the token is saved after every page,
so an interrupted harvest continues where it stopped. Records land in SQLite with an FTS5 index
over title and abstract (``scripts/literature/localindex.py`` searches it).

arXiv asks harvesters to honour ``Retry-After`` on 503 and to leave a few seconds between
requests; this does both.
"""
from __future__ import annotations

import argparse
import http.client
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OAI = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
PAUSE = 3.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY, title TEXT, abstract TEXT, categories TEXT,
    created TEXT, updated TEXT, doi TEXT, authors TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, content='papers', content_rowid='rowid', tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS harvest (
    oai_set TEXT PRIMARY KEY, token TEXT, pages INTEGER, records INTEGER, done INTEGER, updated TEXT);
"""


def fetch(params: dict) -> bytes:
    url = f"{OAI}?{urllib.parse.urlencode(params)}"
    for attempt in range(10):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "FarField-literature-harvest/1.0"}), timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                wait = int(exc.headers.get("Retry-After") or 30)
                print(f"503, waiting {wait}s", flush=True)
                time.sleep(min(wait, 600))
                continue
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            # A connection reset while reading the body is an OSError, not a URLError.
            print(f"{type(exc).__name__}: {exc}; retry {attempt + 1}", flush=True)
            time.sleep(30)
    raise RuntimeError(f"gave up on {url}")


def text(node, path: str) -> str:
    found = node.find(path, NS)
    return " ".join((found.text or "").split()) if found is not None and found.text else ""


def parse(page: bytes) -> tuple[list[tuple], str, bool]:
    root = ET.fromstring(page)
    error = root.find("oai:error", NS)
    if error is not None:
        if error.get("code") == "noRecordsMatch":
            return [], "", True
        raise RuntimeError(f"OAI error {error.get('code')}: {error.text}")
    rows = []
    for record in root.iterfind(".//oai:record", NS):
        meta = record.find("oai:metadata/a:arXiv", NS)
        if meta is None:
            continue
        authors = []
        for author in meta.iterfind("a:authors/a:author", NS):
            name = " ".join(filter(None, (text(author, "a:forenames"), text(author, "a:keyname"))))
            if name:
                authors.append(name)
        rows.append((text(meta, "a:id"), text(meta, "a:title"), text(meta, "a:abstract"),
                     text(meta, "a:categories"), text(meta, "a:created"), text(meta, "a:updated"),
                     text(meta, "a:doi"), "; ".join(authors[:20])))
    token_node = root.find(".//oai:resumptionToken", NS)
    token = (token_node.text or "").strip() if token_node is not None else ""
    return rows, token, not token


def store(db: sqlite3.Connection, rows: list[tuple]) -> None:
    for row in rows:
        old = db.execute("SELECT rowid, title, abstract FROM papers WHERE id=?", (row[0],)).fetchone()
        if old:
            db.execute("INSERT INTO papers_fts(papers_fts, rowid, title, abstract) VALUES('delete', ?, ?, ?)", old)
            db.execute("DELETE FROM papers WHERE id=?", (row[0],))
        cursor = db.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?)", row)
        db.execute("INSERT INTO papers_fts(rowid, title, abstract) VALUES (?,?,?)",
                   (cursor.lastrowid, row[1], row[2]))


def harvest(db: sqlite3.Connection, oai_set: str) -> None:
    state = db.execute("SELECT token, pages, records, done FROM harvest WHERE oai_set=?", (oai_set,)).fetchone()
    if state and state[3]:
        print(f"{oai_set}: already complete ({state[2]} records)", flush=True)
        return
    token, pages, records = (state[0], state[1], state[2]) if state else ("", 0, 0)
    while True:
        params = ({"verb": "ListRecords", "resumptionToken": token} if token else
                  {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": oai_set})
        rows, token, done = parse(fetch(params))
        with db:
            store(db, rows)
            pages, records = pages + 1, records + len(rows)
            db.execute("INSERT OR REPLACE INTO harvest VALUES (?,?,?,?,?,datetime('now'))",
                       (oai_set, token, pages, records, int(done)))
        if pages % 20 == 0 or done:
            print(f"{oai_set}: page {pages}, {records} records{' — complete' if done else ''}", flush=True)
        if done:
            return
        time.sleep(PAUSE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", default=["cs", "stat"])
    args = parser.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    db.execute("PRAGMA journal_mode=DELETE")     # the index lives on NFS: no WAL shared memory
    db.executescript(SCHEMA)
    for oai_set in args.sets:
        harvest(db, oai_set)
    db.execute("INSERT INTO papers_fts(papers_fts) VALUES('optimize')")
    db.commit()
    total = db.execute("SELECT count(*) FROM papers").fetchone()[0]
    print(f"done: {total} papers in {args.db}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
