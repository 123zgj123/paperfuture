"""Multi-source literature search used at mission time.

arXiv, Semantic Scholar, and OpenAlex all run at `fresh` and at survey.
A 429 on one index is `FeedBlocked` for that source, not an empty field.
Admission is a separate tooth (`verifypapers`): CrossRef is a verify
layer, not a third search. Failures stay advisory. Tests pass recorded
JSON and never open a socket.
"""

from __future__ import annotations

import json
import hashlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable
from datetime import datetime, timezone
from html.parser import HTMLParser

from .livefeed import FeedBlocked, FreshWork, as_work

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"
TIMEOUT_S = 20
# One polite retry after a 429/5xx before the source reports FeedBlocked.
# A rate-limited index is a delay, not an empty field; more than one
# retry would stall the mission on a source that is genuinely down.
RETRY_WAIT_S = 2.0
USER_AGENT = (
    "farfield-sources/1.0 (research tool; mailto:farfield-dev@localhost)"
)
Fetcher = Callable[[str], bytes]


class _PaperHTML(HTMLParser):
    """Extract source text and bibliography boundaries; never infer citations."""

    def __init__(self):
        super().__init__()
        self.meta, self.parts, self.heading, self.bibliography = {}, [], [], []
        self.in_heading = False
        self.in_bib = False
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style"}:
            self.skip += 1
        if tag == "meta" and attrs.get("name"):
            self.meta[attrs["name"]] = attrs.get("content", "")
        if tag == "h1":
            self.in_heading = True
        if tag == "li" and "ltx_bibitem" in attrs.get("class", "").split():
            self.in_bib = True
        if self.in_bib and tag == "a":
            self.bibliography.append(attrs.get("href", ""))
        if tag in {"p", "div", "li", "h1", "h2", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)
        if tag == "h1":
            self.in_heading = False
        if tag == "li":
            self.in_bib = False

    def handle_data(self, data):
        if self.skip:
            return
        self.parts.append(data)
        if self.in_heading:
            self.heading.append(data)
        if self.in_bib:
            self.bibliography.append(data)


def read_arxiv_document(work: FreshWork, *, fetcher: Fetcher | None = None) -> dict | FeedBlocked:
    """Read the primary paper, retaining exact text, citation links and hashes.

    This attests source documents, not claims in them. No HTML, mismatched
    identity, or a failed endpoint yields a stand-in document. Extraction
    is intentionally bounded; the omitted text is not claimed to be read.
    """
    from .verifypapers import bare_arxiv, title_overlap, TITLE_OVERLAP
    ident = bare_arxiv(work.arxiv_id)
    if not re.fullmatch(r"\d{4}\.\d{4,5}", ident):
        return FeedBlocked(work.cite_id(), "full_text_reader_requires_arxiv_id")
    documents, parsed = [], {}
    for kind in ("abs", "html"):
        url = f"https://arxiv.org/{kind}/{work.arxiv_id}"
        raw = _get(url, fetcher)
        if isinstance(raw, FeedBlocked):
            return raw
        page = _PaperHTML()
        page.feed(raw.decode("utf-8", "replace"))
        parsed[kind] = page
        documents.append({"url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                          "retrieved_at": datetime.now(timezone.utc).isoformat()})
    meta, full = parsed["abs"].meta, parsed["html"]
    title = meta.get("citation_title", "")
    if (bare_arxiv(meta.get("citation_arxiv_id", "")) != ident or not title
            or work.title and title_overlap(title, work.title) < TITLE_OVERLAP):
        return FeedBlocked(ident, "primary_document_identity_mismatch")
    heading = " ".join(" ".join(full.heading).split())
    if not heading or title_overlap(title, heading) < TITLE_OVERLAP:
        return FeedBlocked(ident, "full_text_identity_or_content_unavailable")
    text = re.sub(r"[ \t]+", " ", "".join(full.parts)).strip()
    refs = re.findall(r"(?:arxiv(?:\.org/(?:abs|pdf|html)/|\s*:\s*))(\d{4}\.\d{4,5})(?:v\d+)?",
                      " ".join(full.bibliography), flags=re.I)
    return {**work.to_dict(), "title": title,
            "abstract": meta.get("citation_abstract") or work.abstract,
            "published": (meta.get("citation_online_date") or meta.get("citation_date") or work.published).replace("/", "-"),
            "first_published": meta.get("citation_date", "").replace("/", "-"),
            "references": list(dict.fromkeys(refs)),
            "full_text_spans": [{"text": text[:16000], "verified": True,
                "url": documents[1]["url"], "source_digest": documents[1]["sha256"],
                "start": 0, "end": min(len(text), 16000), "total_characters": len(text),
                "truncated": len(text) > 16000}],
            "source_documents": documents,
            "epistemic": "SOURCE_DOCUMENT_NOT_EXPERIMENT"}


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def _get(url: str, fetcher: Fetcher | None) -> bytes | FeedBlocked:
    blocked: FeedBlocked | None = None
    for attempt in range(2):
        try:
            return (fetcher or _http_get)(url)
        except urllib.error.HTTPError as error:
            reason = f"HTTP {error.code}"
            retryable = error.code == 429 or error.code >= 500
            if retryable:
                reason += " (retryable)"
            blocked = FeedBlocked(attempted=url, reason=reason)
            if not retryable or attempt:
                return blocked
            time.sleep(RETRY_WAIT_S)
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            return FeedBlocked(attempted=url, reason=str(error))
    return blocked or FeedBlocked(attempted=url, reason="unreachable")


def quoted_query(concepts: tuple[str, ...] | list[str]) -> str:
    """Multiword concepts as quoted phrases for indexes that honor them.

    Joining three phrases into one bag of words is how `code world models
    of executable program state` retrieved drought indexes: every generic
    token matched something highly cited. OpenAlex honors quoted phrases;
    a phrase the author actually wrote is the object, the soup is not.
    """
    parts: list[str] = []
    for concept in concepts:
        text = " ".join(str(concept or "").replace('"', "").split())
        if not text:
            continue
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


def search_scholar(
    query: str,
    *,
    max_results: int = 5,
    fetcher: Fetcher | None = None,
) -> list[FreshWork] | FeedBlocked:
    """Title + abstract search. `fetcher` is the test seam."""
    query = query.strip()
    if not query:
        return FeedBlocked(attempted="scholar", reason="empty query")
    url = S2_API + "?" + urllib.parse.urlencode(
        {
            "query": query,
            "limit": str(max_results),
            "fields": "title,abstract,year,venue,url,externalIds,references.externalIds",
        }
    )
    raw = _get(url, fetcher)
    if isinstance(raw, FeedBlocked):
        return raw
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return FeedBlocked(attempted=url, reason=f"malformed scholar json: {error}")
    works: list[FreshWork] = []
    for row in payload.get("data") or []:
        if not isinstance(row, dict):
            continue
        title = " ".join(str(row.get("title") or "").split())
        if not title:
            continue
        ids = row.get("externalIds") if isinstance(row.get("externalIds"), dict) else {}
        arxiv = str(ids.get("ArXiv") or ids.get("ARXIV") or "").strip()
        doi = str(ids.get("DOI") or "").strip()
        paper_id = str(row.get("paperId") or "").strip()
        year = row.get("year")
        works.append(
            FreshWork(
                title=title,
                published=str(year) if year else "",
                arxiv_id=arxiv,
                abstract=" ".join(str(row.get("abstract") or "").split()),
                source="s2",
                work_id=arxiv or (f"doi:{doi}" if doi else f"s2:{paper_id}"),
                url=str(row.get("url") or ""),
                venue=str(row.get("venue") or ""),
                references=_reference_keys(row),
            )
        )
    return works


def _reference_keys(row: dict[str, Any]) -> tuple[str, ...]:
    """Attested citation ids only. Titles are not invented edges."""
    keys: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        keys.append(text)

    for item in row.get("references") or []:
        if not isinstance(item, dict):
            continue
        ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        arxiv = str(ids.get("ArXiv") or ids.get("ARXIV") or "").strip()
        doi = str(ids.get("DOI") or "").strip()
        paper_id = str(item.get("paperId") or "").strip()
        if arxiv:
            _add(arxiv)
        elif doi:
            _add(f"doi:{doi}")
        elif paper_id:
            _add(f"s2:{paper_id}")
    for item in row.get("referenced_works") or []:
        _add(str(item).rsplit("/", 1)[-1])
    return tuple(keys[:48])


def _union_fresh(keep: FreshWork, other: FreshWork) -> FreshWork:
    refs = tuple(dict.fromkeys([*keep.references, *other.references]))[:48]
    arxiv = keep.arxiv_id or other.arxiv_id
    if refs == keep.references and arxiv == keep.arxiv_id:
        return keep
    return FreshWork(
        title=keep.title,
        published=keep.published,
        arxiv_id=arxiv,
        abstract=keep.abstract,
        source=keep.source,
        work_id=keep.work_id,
        url=keep.url or other.url,
        venue=keep.venue or other.venue,
        references=refs,
    )


def _openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    placed: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                placed.append((pos, str(word)))
    placed.sort()
    return " ".join(word for _, word in placed)


def search_openalex(
    query: str,
    *,
    max_results: int = 5,
    fetcher: Fetcher | None = None,
) -> list[FreshWork] | FeedBlocked:
    query = query.strip()
    if not query:
        return FeedBlocked(attempted="openalex", reason="empty query")
    url = OPENALEX_API + "?" + urllib.parse.urlencode(
        {
            "search": query,
            "per_page": str(max_results),
            "select": "id,display_name,publication_year,doi,abstract_inverted_index,primary_location,referenced_works,ids",
        }
    )
    raw = _get(url, fetcher)
    if isinstance(raw, FeedBlocked):
        return raw
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return FeedBlocked(attempted=url, reason=f"malformed openalex json: {error}")
    return _openalex_works(payload)


def _openalex_works(payload: dict[str, Any]) -> list[FreshWork]:
    """Rows to works. The OpenAlex id is kept as an alias, not discarded."""
    works: list[FreshWork] = []
    for row in payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        title = " ".join(str(row.get("display_name") or "").split())
        if not title:
            continue
        oa_id = str(row.get("id") or "").rsplit("/", 1)[-1]
        doi = str(row.get("doi") or "").replace("https://doi.org/", "").strip()
        loc = row.get("primary_location") if isinstance(row.get("primary_location"), dict) else {}
        source = loc.get("source") if isinstance(loc.get("source"), dict) else {}
        year = row.get("publication_year")
        works.append(
            FreshWork(
                title=title,
                published=str(year) if year else "",
                arxiv_id="",
                abstract=_openalex_abstract(row.get("abstract_inverted_index")),
                source="openalex",
                work_id=f"doi:{doi}" if doi else f"openalex:{oa_id}",
                url=str(row.get("doi") or row.get("id") or ""),
                venue=str(source.get("display_name") or ""),
                references=_reference_keys(row),
            )
        )
        # The OpenAlex id must survive even when the work is keyed by DOI, or a
        # reference recorded as a W-id can never find it.
        works[-1] = _with_alias(works[-1], oa_id, row)
    return works


def _with_alias(work: FreshWork, openalex_id: str, row: dict[str, Any]) -> FreshWork:
    """Keep every identifier the source gave for this work.

    A work keyed by DOI whose OpenAlex id is dropped cannot be found by a
    reference recorded as a W-id, which is how genuine citations failed to
    become edges.
    """
    from dataclasses import replace

    ids = row.get("ids") if isinstance(row.get("ids"), dict) else {}
    aliases = [openalex_id, *[str(value) for value in ids.values() if value]]
    keep = tuple(dict.fromkeys(item for item in aliases if item))
    return replace(work, aliases=keep,
                   url=work.url or f"https://openalex.org/{openalex_id}")


def merge_works(*groups: list[FreshWork]) -> list[FreshWork]:
    """Dedup by cite id, then by normalized title. Keep the richer abstract."""
    by_key: dict[str, FreshWork] = {}
    title_to_key: dict[str, str] = {}
    for group in groups:
        for raw in group or []:
            work = as_work(raw)
            if work is None:
                continue
            title_key = " ".join(work.title.lower().split())
            key = title_to_key.get(title_key) or work.cite_id() or title_key
            current = by_key.get(key)
            if current is None:
                by_key[key] = work
            elif len(work.abstract) > len(current.abstract):
                by_key[key] = _union_fresh(work, current)
            else:
                by_key[key] = _union_fresh(current, work)
            title_to_key[title_key] = key
    return sorted(
        by_key.values(),
        key=lambda w: (w.published, w.title),
        reverse=True,
    )


def fetch_openalex_by_id(
    keys: list[str],
    *,
    fetcher: Fetcher | None = None,
    max_results: int = 25,
) -> list[FreshWork] | FeedBlocked:
    """Dereference canonical identifiers into works.

    This is the step the live path never took: references were kept as strings
    and compared, so a citation the source had already reported could not become
    an edge. OpenAlex resolves its own W-ids, DOIs and arXiv ids in one filtered
    request, so a node's whole reference list costs one call.
    """
    from .workid import ARXIV, DOI, OPENALEX, parse_identifier

    buckets: dict[str, list[str]] = {}
    for key in keys or ():
        parsed = parse_identifier(key)
        if parsed is None:
            continue
        buckets.setdefault(parsed[0], []).append(parsed[1])
    clause = []
    if buckets.get(OPENALEX):
        clause.append("openalex:" + "|".join(buckets[OPENALEX][:50]))
    if buckets.get(DOI):
        clause.append("doi:" + "|".join(buckets[DOI][:50]))
    # An arXiv id has no OpenAlex filter that resolves it directly. Rather than
    # a disabled clause that silently drops those keys, they stay unrequested
    # and show up as the gap between requested and fetched nodes.
    if not clause:
        return []
    works: list[FreshWork] = []
    for filter_clause in clause:
        url = OPENALEX_API + "?" + urllib.parse.urlencode({
            "filter": filter_clause,
            "per-page": str(max_results),
            "select": "id,display_name,publication_year,doi,abstract_inverted_index,"
                      "primary_location,referenced_works,ids",
        })
        raw = _get(url, fetcher)
        if isinstance(raw, FeedBlocked):
            return raw
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return FeedBlocked(attempted=url, reason=f"malformed openalex json: {error}")
        works.extend(_openalex_works(payload))
    return works


def fetch_openalex_citing(
    keys: list[str],
    *,
    fetcher: Fetcher | None = None,
    max_results: int = 8,
) -> list[FreshWork] | FeedBlocked:
    """Later works that cite this one. Forward provenance, bounded per call.

    OpenAlex's ``cites:`` filter takes W-ids and nothing else. Most works here
    are canonically keyed by DOI, so requiring a W-id made this return an empty
    list for almost every node — indistinguishable from "nobody cites it", and
    the reason no forward edge ever appeared in a graph. Keys in another
    namespace are resolved to W-ids first, in one batched request.
    """
    from .workid import OPENALEX, parse_identifier

    targets: list[str] = []
    unresolved: list[str] = []
    for key in keys or ():
        parsed = parse_identifier(key)
        if parsed is None:
            continue
        (targets if parsed[0] == OPENALEX else unresolved).append(
            parsed[1] if parsed[0] == OPENALEX else key)
    if unresolved:
        resolved = fetch_openalex_by_id(unresolved, fetcher=fetcher,
                                        max_results=max(len(unresolved), 10))
        if isinstance(resolved, FeedBlocked):
            return resolved
        for work in resolved:
            for alias in work.aliases:
                parsed = parse_identifier(str(alias))
                if parsed and parsed[0] == OPENALEX:
                    targets.append(parsed[1])
                    break
    if not targets:
        return []
    url = OPENALEX_API + "?" + urllib.parse.urlencode({
        "filter": "cites:" + "|".join(targets[:10]),
        "per-page": str(max_results),
        "select": "id,display_name,publication_year,doi,abstract_inverted_index,"
                  "primary_location,referenced_works,ids",
    })
    raw = _get(url, fetcher)
    if isinstance(raw, FeedBlocked):
        return raw
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return FeedBlocked(attempted=url, reason=f"malformed openalex json: {error}")
    return _openalex_works(payload)
