"""Versioned collection, frozen-snapshot retrieval and trajectory coverage."""
from __future__ import annotations

from collections import Counter
import datetime as dt
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .config import DOMAINS
from .contracts import ContractError, require_refs
from .storage import digest, seal, timestamp, unseal


def paper_record(raw: dict) -> dict:
    required = {"paper_id", "version_id", "title", "first_public_at", "version_public_at", "source_url", "sections"}
    if not required <= set(raw):
        raise ValueError(f"paper is missing fields: {sorted(required-set(raw))}")
    for key in required - {"sections"}:
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"paper.{key} must be a nonempty string")
    if "@" in raw["paper_id"] or "@" in raw["version_id"]:
        raise ValueError("paper_id and version_id cannot contain @ (the version-key separator)")
    first, version = timestamp(raw["first_public_at"]), timestamp(raw["version_public_at"])
    if first > version:
        raise ValueError("first_public_at cannot follow version_public_at")
    if not isinstance(raw["sections"], dict) or not raw["sections"]:
        raise ValueError("paper.sections must contain at least abstract or full-text sections")
    if any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in raw["sections"].items()):
        raise ValueError("paper section names/text must be nonempty strings")
    references = raw.get("references", [])
    if not isinstance(references, list) or any(not isinstance(x, str) for x in references):
        raise ValueError("references must be actual cited paper identifiers, not similarity neighbors")
    paper = {k: raw[k] for k in sorted(required)}
    paper["references"] = sorted(set(references))
    paper["temporal_provenance"] = raw.get("temporal_provenance", "importer_attested_version_dates")
    paper["content_level"] = "abstract_only" if set(paper["sections"]) == {"abstract"} else "provided_sections"
    paper["key"] = paper["paper_id"] + "@" + paper["version_id"]
    paper["content_hash"] = digest(paper)
    return paper


class LiteratureStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS versions (key TEXT PRIMARY KEY, paper_id TEXT, content_hash TEXT, record TEXT)")
        self.db.commit()

    def close(self):
        self.db.close()

    def add(self, raw: dict):
        paper = paper_record(raw)
        # An abstract-only observation can later acquire additional sections.
        # Keep BOTH representations immutable; existing snapshots never change.
        storage_key = paper["key"] + "#" + paper["content_hash"]
        with self.db:
            # Serialize check+append across DIFFERENT runs sharing this library.
            # A per-run lock alone cannot prevent conflicting version imports.
            self.db.execute("BEGIN IMMEDIATE")
            records = self.db.execute("SELECT record FROM versions WHERE paper_id=?", (paper["paper_id"],)).fetchall()
            for (record,) in records:
                old = json.loads(record)
                if old["key"] != paper["key"]:
                    continue
                fixed = set(paper) - {"sections", "content_level", "content_hash", "references"}
                changed = any(old[k] != paper[k] for k in fixed)
                changed |= any(old["sections"][k] != paper["sections"][k] for k in old["sections"].keys() & paper["sections"].keys())
                if changed:
                    raise ValueError(f"immutable version changed: {paper['key']}; use a distinct audited version identifier")
            self.db.execute("INSERT OR IGNORE INTO versions VALUES(?,?,?,?)",
                            (storage_key, paper["paper_id"], paper["content_hash"], json.dumps(paper, ensure_ascii=False)))
        return storage_key

    def freeze(self, keys: list[str], cutoff: str) -> "Snapshot":
        members = {}
        excluded = []
        limit = timestamp(cutoff)
        for key in sorted(set(keys)):
            row = self.db.execute("SELECT record FROM versions WHERE key=?", (key,)).fetchone()
            if row is None:
                raise ValueError(f"missing collected version {key}")
            paper = json.loads(row[0])
            if timestamp(paper["first_public_at"]) > limit or timestamp(paper["version_public_at"]) > limit:
                excluded.append(key)
                continue
            old = members.get(paper["paper_id"])
            def preference(p):
                return (timestamp(p["version_public_at"]), sum(len(s) for s in p["sections"].values()), p["content_hash"])
            if old is None or preference(paper) > preference(old):
                members[paper["paper_id"]] = paper
        if len(members) < 2:
            raise ValueError("fewer than two eligible papers before cutoff; collect more evidence")
        payload = {"cutoff": cutoff, "papers": sorted(members.values(), key=lambda p: p["key"]), "excluded_future_versions": excluded}
        return Snapshot(payload)


class Snapshot:
    """The only lookup surface used downstream. No access to future versions."""
    def __init__(self, payload: dict):
        self.payload = payload
        self.id = digest(payload)
        self.papers = {p["key"]: p for p in payload["papers"]}
        if len(self.papers) != len(payload["papers"]):
            raise ValueError("duplicate snapshot versions")
        self.evidence = {}
        for paper in self.papers.values():
            if paper_record(paper)["content_hash"] != paper["content_hash"]:
                raise ValueError("paper content hash mismatch")
            if timestamp(paper["version_public_at"]) > timestamp(payload["cutoff"]):
                raise ValueError("future paper in frozen snapshot")
            for section, text in paper["sections"].items():
                # Character offsets and exact quotes are generated by code, not LLM.
                for start in range(0, len(text), 1400):
                    end = min(start + 1400, len(text))
                    quote = text[start:end]
                    span = {"paper_key": paper["key"], "section": section, "start": start, "end": end,
                            "quote": quote, "quote_hash": digest(quote)}
                    self.evidence["ev_" + digest(span)[:24]] = span

    def get(self, key):
        if key not in self.papers:
            raise ValueError(f"paper is outside the frozen snapshot: {key}")
        return self.papers[key]

    def spans(self, key):
        self.get(key)
        return {k: v for k, v in self.evidence.items() if v["paper_key"] == key}

    def references(self, key):
        identifiers = set(self.get(key)["references"])
        return [p["key"] for p in self.papers.values() if p["paper_id"] in identifiers or p["key"] in identifiers]


ATOM = {"a": "http://www.w3.org/2005/Atom"}


def parse_arxiv(raw: bytes) -> list[dict]:
    root = ET.fromstring(raw)
    papers = []
    for entry in root.findall("a:entry", ATOM):
        def value(name):
            return " ".join((entry.findtext(f"a:{name}", "", ATOM)).split())
        identifier = value("id").split("/abs/")[-1]
        match = re.fullmatch(r"(.+)v(\d+)", identifier)
        if not match:
            raise ValueError("arxiv returned an error or an unversioned record; refusing ambiguous metadata")
        papers.append({"paper_id": "arxiv:" + match[1], "version_id": "v" + match[2], "title": value("title"),
                       "first_public_at": value("published"), "version_public_at": value("updated"),
                       "source_url": "https://arxiv.org/abs/" + identifier,
                       "sections": {"abstract": value("summary")}, "references": [],
                       "temporal_provenance": "arxiv_version_specific_atom_timestamp"})
    return papers


def collect_arxiv(source: dict, config: dict, cache: Path) -> list[dict]:
    """Search by submitted time, then retrieve explicit v1 metadata.

    Version API: https://info.arxiv.org/help/api/user-manual.html#51-details-of-query-construction
    Search metadata is only discovery input; only explicit v1 text is admitted.
    """
    cache.mkdir(parents=True, exist_ok=True)
    last_request = [0.0]
    def fetch(params):
        path = cache / (digest(params) + ".json")
        if path.exists():
            return unseal(path).encode()
        if config["api_mode"] == "replay":
            raise RuntimeError("arxiv replay cache miss")
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        for attempt in range(3):
            time.sleep(max(0, 3.1 - (time.monotonic() - last_request[0])))
            last_request[0] = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "PaperFuture/0.2 (research metadata collection)"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                ET.fromstring(raw)  # never cache an HTML gateway error
                seal(path, raw.decode())
                return raw
            except (OSError, ET.ParseError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    queries = source.get("queries", DOMAINS[config["domain"]]["queries"])
    maximum = int(source.get("max_results", 500))
    page_size = min(200, int(source.get("page_size", 100)))
    if maximum < 1 or page_size < 1 or not queries:
        raise ValueError("arxiv collection needs positive limits and queries")
    end = timestamp(config["cutoff"]).strftime("%Y%m%d%H%M")
    start = source.get("start_date", "20000101")
    if not re.fullmatch(r"\d{8}", start):
        raise ValueError("arxiv start_date must be YYYYMMDD")
    identifiers = set()
    for identifier in source.get("ids", []):
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v1)?", identifier):
            raise ValueError("explicit arxiv IDs must be unversioned or v1 identifiers")
        identifiers.add(identifier if identifier.endswith("v1") else identifier + "v1")
    for query in ([] if source.get("ids") else queries):
        for offset in range(0, maximum, page_size):
            amount = min(page_size, maximum-offset)
            params = {"search_query": f"({query}) AND submittedDate:[{start}0000 TO {end}]",
                      "start": offset, "max_results": amount, "sortBy": "submittedDate", "sortOrder": "ascending"}
            rows = parse_arxiv(fetch(params))
            identifiers.update(p["paper_id"].removeprefix("arxiv:") + "v1" for p in rows)
            if len(rows) < amount:
                break
    result = []
    ids = sorted(identifiers)
    for offset in range(0, len(ids), 50):
        requested = ids[offset:offset+50]
        rows = parse_arxiv(fetch({"id_list": ",".join(requested), "max_results": len(requested)}))
        returned = [p["paper_id"].removeprefix("arxiv:") + p["version_id"] for p in rows]
        if set(returned) != set(requested):
            raise ValueError("arxiv version response does not match explicitly requested v1 identifiers")
        result.extend(rows)
    if source.get("full_text", "none") == "pdf":
        if shutil.which("pdftotext") is None:
            raise ValueError("full_text=pdf requires the local pdftotext binary (Poppler); no GPU or model weights are needed")
        for paper in result:
            identifier = paper["paper_id"].removeprefix("arxiv:") + paper["version_id"]
            text_path = cache / (digest({"pdf": identifier, "extractor": "pdftotext-layout-v1"}) + ".json")
            if text_path.exists():
                extracted = unseal(text_path)
            else:
                if config["api_mode"] == "replay":
                    raise RuntimeError("versioned full-text replay cache miss")
                time.sleep(max(0, 3.1 - (time.monotonic() - last_request[0])))
                last_request[0] = time.monotonic()
                request = urllib.request.Request("https://arxiv.org/pdf/" + identifier,
                    headers={"User-Agent": "PaperFuture/0.2 (version-specific research full text)"})
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        raw = response.read(30*1024*1024 + 1)
                    if len(raw) > 30*1024*1024 or not raw.startswith(b"%PDF"):
                        raise ValueError("source did not return a PDF within the 30 MB limit")
                    with tempfile.TemporaryDirectory(prefix="paperfuture-pdf-") as temporary:
                        pdf, out = Path(temporary)/"paper.pdf", Path(temporary)/"paper.txt"
                        pdf.write_bytes(raw)
                        subprocess.run(["pdftotext", "-layout", str(pdf), str(out)], check=True, capture_output=True, timeout=60)
                        extracted = out.read_text(encoding="utf-8").strip()
                    if len(extracted) < 200:
                        raise ValueError("insufficient extractable PDF text; supply audited text sections manually")
                    seal(text_path, extracted)
                except (OSError, ValueError, subprocess.SubprocessError):
                    if source.get("require_full_text", False):
                        raise
                    # Missing full text is visible through content_level, not a fabricated body.
                    continue
            paper["sections"]["full_text"] = extracted
    return result


def collect(config: dict, store: LiteratureStore, cache: Path) -> list[str]:
    keys = []
    for source in config["sources"]:
        if source["type"] == "jsonl":
            with Path(source["path"]).open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            keys.append(store.add(json.loads(line)))
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f"{source['path']}:{line_number}: {exc}") from exc
        else:
            keys.extend(store.add(p) for p in collect_arxiv(source, config, cache))
    return keys


def tokens(text):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def bm25(query: str, documents: dict[str, str]) -> list[str]:
    """Okapi BM25 over the entire eligible snapshot (k1=1.5, b=.75)."""
    counts = {k: Counter(tokens(v)) for k, v in documents.items()}
    lengths = {k: sum(v.values()) for k, v in counts.items()}
    average = sum(lengths.values()) / max(1, len(lengths)) or 1
    terms = set(tokens(query))
    df = {term: sum(term in row for row in counts.values()) for term in terms}
    scores = {}
    for key, row in counts.items():
        score = 0.0
        for term in terms:
            tf = row.get(term, 0)
            idf = math.log(1 + (len(counts)-df[term]+0.5)/(df[term]+0.5))
            score += idf * tf * 2.5 / (tf + 1.5*(0.25+0.75*lengths[key]/average))
        scores[key] = score
    return sorted(scores, key=lambda k: (-scores[k], k))


def cosine(a, b):
    if len(a) != len(b):
        raise ContractError("embedding dimension changed between document/query batches")
    return sum(x*y for x,y in zip(a,b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))


def rrf(rankings: list[list[str]], k=60) -> list[str]:
    scores = Counter()
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] += 1/(k+rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


def select_corpus(snapshot: Snapshot, config: dict, gateway) -> dict:
    options = config["retrieval"]
    documents = {key: p["title"] + "\n" + p["sections"].get("abstract", next(iter(p["sections"].values())))
                 for key,p in snapshot.papers.items()}
    # Document embeddings see every eligible document, independently of BM25.
    rankings = [bm25(config["question"], documents)[:options["recall_k"]]]
    vectors = {}
    keys = sorted(documents)
    if options["method"] == "hybrid":
        for offset in range(0, len(keys), 32):
            batch = keys[offset:offset+32]
            embedding = gateway.embeddings([documents[k] for k in batch], identity=f"documents:{snapshot.id}:{offset}")
            vectors.update(zip(batch, embedding))
        query = gateway.embeddings([config["question"]], identity=f"query:{snapshot.id}")[0]
        rankings.append(sorted(keys, key=lambda k: (-cosine(query, vectors[k]), k))[:options["recall_k"]])
    candidates = rrf(rankings, options["rrf_k"])[:options["candidate_limit"]]
    reranked = {}
    for offset in range(0, len(candidates), options["rerank_batch"]):
        batch = candidates[offset:offset+options["rerank_batch"]]
        def check(result):
            actual = [x["paper_key"] for x in result["results"]]
            if len(actual) != len(batch) or set(actual) != set(batch):
                raise ContractError("reranker must score every supplied candidate exactly once")
        result = gateway.structured("rerank", {"question": config["question"], "domain": DOMAINS[config["domain"]],
            "papers": [{"paper_key": k, "text": documents[k]} for k in batch]},
            "Score relevance 0 (unrelated) through 4 (directly relevant). Include historically enabling work. Return every input identifier exactly once. Do not use current citation counts.",
            identity=f"rerank:{snapshot.id}:{offset}", check=check)
        reranked.update({r["paper_key"]: r for r in result["results"]})
    extractions = {}
    for key in candidates:
        if reranked[key]["relevance"] < options["min_relevance"]:
            continue
        paper = snapshot.get(key)
        spans = snapshot.spans(key)
        def check(result):
            require_refs(result["evidence_ids"], spans, "paper extraction", nonempty=True)
        metadata = {k:v for k,v in paper.items() if k != "sections"}
        extraction = gateway.structured("extract", {"question": config["question"], "paper": metadata, "evidence_spans": spans},
            "Extract the technical problem, mechanism, assumptions, demonstrated capabilities, limitations and unresolved bottlenecks. Cite 1 to 8 most informative supplied evidence IDs supporting these fields, prioritizing method, demonstrated capability and limitations. Abstract-only papers cannot support unreported technical details. Quality measures evidence specificity (0-4), not venue or citations. Subtopics should be reusable short technical labels. Mark missing content in limitations.",
            identity=f"extract:{paper['content_hash']}", check=check)
        if extraction["relevance"] >= options["min_relevance"] and extraction["quality"] >= options["min_quality"]:
            extractions[key] = extraction
    if len(extractions) < 2:
        raise ValueError("fewer than two relevant, evidence-bearing papers after screening; broaden collection or audit the relevance judgments")
    def features(key):
        paper, ext = snapshot.get(key), extractions[key]
        year = timestamp(paper["version_public_at"]).year
        return {"year:"+str(year), *("topic:"+s.lower() for s in ext["subtopics"]),
                "mechanism:"+" ".join(tokens(ext["mechanism"])[:8])}
    covered, selected, reasons = set(), [], {}
    remaining = set(extractions)
    while remaining and len(selected) < options["selected_limit"]:
        scores = {}
        for key in remaining:
            ext = extractions[key]
            f = features(key)
            gain = len(f-covered)/max(1,len(f))
            if vectors:
                redundancy = max((max(0, cosine(vectors[key], vectors[s])) for s in selected), default=0)
            else:
                bag = set(tokens(documents[key]))
                redundancy = max((len(bag & set(tokens(documents[s]))) / max(1,len(bag | set(tokens(documents[s])))) for s in selected), default=0)
            base = (reranked[key]["relevance"] + ext["relevance"] + ext["quality"]) / 12
            scores[key] = base + options["coverage_weight"]*gain - options["redundancy_weight"]*redundancy
        key = min(remaining, key=lambda k: (-scores[k], k))
        reasons[key] = {"selection_score": scores[key], "new_coverage": sorted(features(key)-covered),
                        "rationale": extractions[key]["selection_reason"]}
        selected.append(key)
        remaining.remove(key)
        covered.update(features(key))
    eligible_features = set().union(*(features(k) for k in extractions))
    return {"snapshot_id": snapshot.id, "selected_keys": selected,
            "extractions": {k: extractions[k] for k in selected}, "selection_reasons": reasons,
            "coverage": sorted(covered), "coverage_gaps": sorted(eligible_features-covered),
            "candidate_count": len(candidates), "eligible_count": len(extractions), "snapshot_count": len(snapshot.papers),
            "retrieval_method": options["method"], "rankings": rankings, "rerank": reranked,
            "limitations": ["Coverage is measured within the collected candidate pool, not the entire research field.",
                            "Abstract-only evidence limits judgments of mechanism and verified capability."]}
