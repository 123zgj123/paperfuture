"""Year-stratified, resumable literature acquisition. All bytes live in data_root.

The manifest measures coverage of declared queries, never universal field recall.
Discovery metadata is not admitted as historical evidence: explicit versions are.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .literature import LiteratureStore, parse_arxiv, paper_record
from .storage import atomic_json, atomic_text, digest, exclusive, now, read_json, seal, timestamp, unseal


PLANS = {
    "agent_security": {"start_year": 2015, "facets": {
        "prompt_injection": '(ti:"prompt injection" OR abs:"prompt injection" OR ti:"indirect prompt")',
        "agent_authorization": '(all:"language model" OR all:LLM) AND (all:"agent security" OR all:"tool permissions" OR all:"access control" OR all:"least privilege")',
        "tool_safety": '(all:"language model" OR all:LLM) AND (all:"tool use" OR all:"tool-use" OR all:"tool-integrated") AND (all:security OR all:safety OR all:attack)',
        "adaptive_attacks": '(all:"language model" OR all:LLM) AND (all:jailbreak OR all:jailbreaking OR all:"adaptive attack")',
        "historical_foundations": '(all:"adversarial" AND (all:"text classification" OR all:"natural language processing"))',
    }},
    "multimodal_self_evolution_xai": {"start_year": 2015, "facets": {
        "self_improvement": '(all:multimodal OR all:"vision language" OR all:"vision-language") AND (all:"self improvement" OR all:"self-improvement" OR all:"self evolution" OR all:"self-evolving" OR all:"self refinement")',
        "explanation_faithfulness": '(all:multimodal OR all:"vision language" OR all:"vision-language") AND (all:interpretability OR all:explainability OR all:"mechanistic interpretation" OR all:"causal tracing")',
        "feedback_and_reflection": '(all:multimodal OR all:"vision language" OR all:"vision-language") AND (all:"self critique" OR all:"self-correction" OR all:"self reflection" OR all:"reward hacking")',
        "representation_and_attribution": '(all:multimodal OR all:"vision language" OR all:"vision-language") AND (all:attribution OR all:"representation evolution" OR all:"concept bottleneck")',
        "historical_foundations": '(all:"visual question answering" AND (all:explanation OR all:reasoning))',
    }},
    "ai4eda": {"start_year": 2010, "facets": {
        "rtl_generation": '(all:Verilog OR all:"RTL generation" OR all:SystemVerilog) AND (all:"language model" OR all:learning OR all:neural)',
        "formal_verification": '(all:"hardware verification" OR all:"RTL verification" OR all:"assertion generation") AND (all:learning OR all:"language model" OR all:neural)',
        "physical_design": '(all:"circuit placement" OR all:"chip placement" OR all:"physical design") AND (all:"machine learning" OR all:"reinforcement learning" OR all:"deep learning")',
        "logic_synthesis": '(all:"logic synthesis" OR all:"high level synthesis") AND (all:learning OR all:"language model" OR all:neural)',
        "design_optimization": '(all:"electronic design automation" OR all:"power performance area") AND (all:"machine learning" OR all:"language model" OR all:"deep learning")',
    }},
}


def data_root() -> Path:
    """An explicit data volume, with no silent fallback to the system volume."""
    configured = os.environ.get("PAPERFUTURE_DATA_ROOT")
    root = Path(configured) if configured else Path("/mnt/efs/zhangguijia/paperfuture")
    if not configured and not Path("/mnt/efs").is_mount():
        raise ValueError("Set PAPERFUTURE_DATA_ROOT to a mounted data volume; no system-disk fallback is allowed")
    return root.resolve()


class Archive:
    def __init__(self, root: Path, *, replay=False):
        self.root, self.replay = root, replay
        root.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str, *, pdf=False) -> tuple[bytes, Path]:
        path = self.root / (digest(url) + (".pdf" if pdf else ".xml"))
        receipt = path.with_suffix(path.suffix + ".json")
        if path.exists() and receipt.exists():
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != read_json(receipt)["sha256"]:
                raise ValueError(f"source archive hash mismatch: {path}")
            return raw, path
        if self.replay:
            raise RuntimeError("source archive replay cache miss")
        # All arXiv metadata/PDF downloads sharing this archive obey one lock/rate.
        with exclusive(self.root / ".request.lock", blocking=True):
            if path.exists() and receipt.exists():
                raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=read_json(receipt)["sha256"]:
                    raise ValueError(f"source archive hash mismatch: {path}")
                return raw,path
            for attempt in range(3):
                stamp = self.root / ".last_request"
                previous = float(stamp.read_text()) if stamp.exists() else 0
                time.sleep(max(0, 3.1 - (time.time() - previous)))
                stamp.write_text(str(time.time()))
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "PaperFuture/0.4 research-literature-archive"})
                    limit = (128 if pdf else 40)*1024*1024
                    with urllib.request.urlopen(req, timeout=60) as response:
                        raw = response.read(limit+1)
                    if len(raw) > limit:
                        raise ValueError(f"source exceeds the declared {limit//(1024*1024)} MB per-document limit")
                    if pdf and not raw.startswith(b"%PDF"):
                        raise ValueError("source returned non-PDF content")
                    if not pdf:
                        ET.fromstring(raw)
                    temporary = path.with_suffix(path.suffix + ".partial")
                    with temporary.open("wb") as stream:
                        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                    os.replace(temporary, path)
                    atomic_json(receipt, {"url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                                          "bytes": len(raw), "retrieved_at": now()})
                    return raw, path
                except (OSError, ValueError, ET.ParseError) as exc:
                    if isinstance(exc, ValueError) and str(exc).startswith("source exceeds"):
                        raise  # A fixed size limit cannot be repaired by retrying.
                    if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                        raise
                    if attempt == 2:
                        raise
                    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                        # A rate limit is unrelated to window size. Respect a
                        # bounded Retry-After delay before the next attempt.
                        retry_after = (exc.headers or {}).get("Retry-After", "30")
                        delay = float(retry_after) if str(retry_after).isdigit() else 30
                        time.sleep(min(60, max(30, delay)) * (attempt+1))
                    else:
                        time.sleep(4 * (attempt+1))
        raise AssertionError("unreachable")

    def query(self, params):
        return self.fetch("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params))


def enrich_fulltext(paper: dict, archive: Archive) -> dict:
    """Retain version-specific PDF bytes, extracted text, hashes and bibliography."""
    if not paper["paper_id"].startswith("arxiv:"):
        return paper
    identifier = paper["paper_id"][6:] + paper["version_id"]
    raw, pdf = archive.fetch("https://arxiv.org/pdf/" + identifier, pdf=True)
    text_path = pdf.with_suffix(".txt")
    if not text_path.exists():
        out = text_path.with_suffix(".txt.partial")
        subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), str(out)], check=True, capture_output=True, timeout=90)
        text = out.read_text(encoding="utf-8").strip()
        out.unlink(missing_ok=True)
        if len(text) < 300:
            raise ValueError("PDF does not contain sufficient extractable text")
        atomic_text(text_path, text)
    text = text_path.read_text(encoding="utf-8")
    # IDs found in the bibliography are citations, never inferred neighbors.
    headings = list(re.finditer(r"(?im)^\s*(?:\d+\s+)?(?:references|bibliography)\s*$", text))
    bibliography = text[headings[-1].end():] if headings else ""
    refs = {"arxiv:" + m for m in re.findall(r"(?:arXiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})", bibliography, flags=re.I)}
    refs.discard(paper["paper_id"])
    augmented = dict(paper)
    augmented["sections"] = {**paper["sections"], "full_text": text}
    augmented["references"] = sorted(set(paper.get("references", [])) | refs)
    augmented["artifacts"] = {"pdf": str(pdf), "pdf_sha256": hashlib.sha256(raw).hexdigest(),
                              "text": str(text_path), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                              "reference_extraction": "arxiv_identifiers_in_detected_bibliography; incomplete"}
    return augmented


def discover_window(archive, query, begin, finish, record_page, *, offset=0,
                    known_total=None, initial_count=0, depth=0):
    """Recover expensive/failed arXiv windows by bisecting disjoint date ranges.

    Parent receipts and errors remain visible. Only complete children can resolve
    a failed parent; replay never fetches missing responses from the network.
    """
    window = {"start": begin.isoformat(), "end": finish.isoformat(),
              "retrieved": initial_count, "matched": known_total, "error": None}
    try:
        while window["matched"] is None or offset < window["matched"]:
            params = {"search_query": f'({query}) AND submittedDate:[{begin.strftime("%Y%m%d%H%M")} TO {finish.strftime("%Y%m%d%H%M")}]',
                      "start": offset, "max_results": 200,
                      "sortBy": "submittedDate", "sortOrder": "descending"}
            raw, source_path = archive.query(params)
            window["matched"] = int(ET.fromstring(raw).findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults", "0"))
            rows = record_page(raw, source_path)
            window["retrieved"] += len(rows)
            offset += len(rows)
            if offset >= window["matched"]:
                break
            if not rows or offset >= 20000:
                raise ValueError("incomplete source page or deep pagination limit")
    except (OSError, ValueError, RuntimeError, ET.ParseError) as exc:
        failure = type(exc).__name__ + str(getattr(exc, "code", ""))
        window["error"] = failure
        minutes = int((finish-begin).total_seconds() // 60)
        rate_limited = isinstance(exc, urllib.error.HTTPError) and exc.code == 429
        if not rate_limited and minutes >= 1440 and depth < 9:
            midpoint = begin + dt.timedelta(minutes=minutes//2)
            children = [discover_window(archive, query, left, right, record_page, depth=depth+1)
                        for left, right in ((begin, midpoint), (midpoint+dt.timedelta(minutes=1), finish))]
            window.update(initial_error=failure, children=children)
            # Counts describe the disjoint children, not partially fetched pages
            # plus the same papers fetched a second time during recovery.
            window["retrieved"] = sum(c["retrieved"] for c in children)
            if all(c["matched"] is not None for c in children):
                recovered_total = sum(c["matched"] for c in children)
                if window["matched"] is not None:
                    window["initial_matched"] = window["matched"]
                window["matched"] = recovered_total
            if all(c["error"] is None and c["retrieved"] == c["matched"] for c in children):
                window["error"] = None
                window["recovered_by_splitting"] = True
    return window


def harvest(domain: str, root: Path, cutoff: str, *, historical_limit=400, recent_limit=1600,
            fulltext_limit=100, replay=False, progress=print) -> dict:
    plan = PLANS[domain]
    end = timestamp(cutoff)
    directory = root / "datasets" / domain / end.strftime("%Y%m%d")
    directory.mkdir(parents=True, exist_ok=True)
    archive = Archive(root / "cache" / "arxiv", replay=replay)
    # A per-domain lock prevents two harvests from replacing one manifest/export.
    with exclusive(directory / ".lock"):
        return _harvest(domain, plan, directory, archive, cutoff, historical_limit,
                        recent_limit, fulltext_limit, progress)


def _harvest(domain, plan, directory, archive, cutoff, historical_limit, recent_limit, fulltext_limit, progress):
    end = timestamp(cutoff)
    checkpoint = directory / "discovery.json"
    discovery = read_json(checkpoint) if checkpoint.exists() else {"strata": [], "identifiers": {}}
    specification = {"domain": domain, "cutoff": cutoff, "queries": plan, "historical_limit": historical_limit,
                     "recent_limit": recent_limit, "recent_years": [end.year-1, end.year], "version_policy": "v1"}
    seal(directory / "spec.json", specification)
    done = {s["year"] for s in discovery["strata"] if s["status"] == "complete" and not s["truncated"]}
    # Newest years first; each year receives its own independent retrieval budget.
    for year in range(end.year, plan["start_year"]-1, -1):
        if year in done:
            continue
        cap = recent_limit if year >= end.year-1 else historical_limit
        query = " OR ".join("("+q+")" for facet,q in plan["facets"].items()
                            if facet != "historical_foundations" or year < 2023)
        until = min(end, dt.datetime(year, 12, 31, 23, 59, tzinfo=dt.timezone.utc))
        dated = f'({query}) AND submittedDate:[{year}01010000 TO {until.strftime("%Y%m%d%H%M")}]'
        found, total, error = set(), None, None
        windows=[]
        def record_page(raw,source_path):
            rows=parse_arxiv(raw)
            for row in rows:
                identifier=row["paper_id"][6:]+"v1"
                found.add(identifier)
                discovery["identifiers"][identifier]={"discovered_year":year,"source":str(source_path)}
            return rows
        initial_error, initial_count = None, 0
        try:
            first_params={"search_query":dated,"start":0,"max_results":200,
                          "sortBy":"submittedDate","sortOrder":"descending"}
            raw,source_path=archive.query(first_params)
            total=int(ET.fromstring(raw).findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults","0"))
            initial_count=len(record_page(raw,source_path))
        except (OSError, ValueError, RuntimeError, ET.ParseError) as exc:
            initial_error = type(exc).__name__+str(getattr(exc,"code",""))
        if total is None or total>800:
            # A failed year query is also recoverable through narrower windows.
            import calendar
            for month in range(1,13):
                begin=dt.datetime(year,month,1,tzinfo=dt.timezone.utc)
                if begin>end: break
                finish=min(end,dt.datetime(year,month,calendar.monthrange(year,month)[1],23,59,tzinfo=dt.timezone.utc))
                windows.append(discover_window(archive,query,begin,finish,record_page))
                atomic_json(checkpoint,discovery)
            if total is None and all(w["matched"] is not None for w in windows):
                total=sum(w["matched"] for w in windows)
        else:
            windows.append(discover_window(archive,query,dt.datetime(year,1,1,tzinfo=dt.timezone.utc),until,
                                           record_page,offset=initial_count,known_total=total,initial_count=initial_count))
        failed=[w["error"] for w in windows if w["error"]]
        error=", ".join(sorted(set(failed))) if failed else None
        stratum = {"year": year, "status": "failed" if error else "complete", "error": error,
                   "matched": total, "retrieved": len(found), "limit": cap,
                   "truncated": total is not None and total > len(found), "query": dated,"windows":windows,
                   "initial_error": initial_error}
        discovery["strata"] = [s for s in discovery["strata"] if s["year"] != year] + [stratum]
        atomic_json(checkpoint, discovery)
        progress(f"{domain} {year}: {len(found)}/{total} discovered" + (f" ({error})" if error else ""), flush=True)
    records_path = directory / "version_records.json"
    records = read_json(records_path) if records_path.exists() else {}
    missing = sorted(set(discovery["identifiers"]) - set(records))
    failures = []
    for offset in range(0, len(missing), 50):
        batch = missing[offset:offset+50]
        try:
            raw, _ = archive.query({"id_list": ",".join(batch), "max_results": len(batch)})
            rows = parse_arxiv(raw)
            returned = {p["paper_id"][6:]+p["version_id"] for p in rows}
            if returned != set(batch):
                raise ValueError("explicit version response mismatch")
            for p in rows:
                if timestamp(p["version_public_at"]) <= end and timestamp(p["first_public_at"]) <= end:
                    records[p["paper_id"][6:]+p["version_id"]] = p
            atomic_json(records_path, records)
        except (OSError, ValueError, RuntimeError, ET.ParseError) as exc:
            failures.append({"ids": batch, "error": type(exc).__name__})
        progress(f"{domain}: {len(records)} version-frozen metadata records", flush=True)
    # Detailed full-text sample: guarantee historical years, then emphasize recency.
    by_year = {}
    for key, p in records.items():
        by_year.setdefault(timestamp(p["first_public_at"]).year, []).append(key)
    from .literature import bm25,rrf
    chosen = [];fulltext_selection={}
    for year in sorted(by_year):
        if year < end.year-1:
            docs={k:records[k]["title"]+" "+records[k]["sections"]["abstract"] for k in by_year[year]}
            routes={facet:bm25(query,docs,positive_only=True) for facet,query in plan["facets"].items()}
            # Relevance and facet diversity, never lexicographic arXiv IDs.
            ranked=rrf(list(routes.values()));leaders=[r[0] for r in routes.values() if r]
            selected=list(dict.fromkeys(leaders+ranked))[:min(3,len(docs))]
            chosen.extend(selected)
            fulltext_selection.update({k:{"year":year,"routes":[f for f,r in routes.items() if k in r[:3]]} for k in selected})
    recent_years=[y for y in (end.year-1,end.year) if y in by_year]
    for i,year in enumerate(recent_years):
        quota=max(0,(fulltext_limit-len(chosen))//(len(recent_years)-i))
        docs={k:records[k]["title"]+" "+records[k]["sections"]["abstract"] for k in by_year[year]}
        rankings=[bm25(q,docs) for facet,q in plan["facets"].items() if facet!="historical_foundations"]
        selected=[]
        for rank in range(max((len(r) for r in rankings),default=0)):
            for ranking in rankings:
                if rank<len(ranking) and ranking[rank] not in selected and len(selected)<quota:
                    selected.append(ranking[rank])
            if len(selected)>=quota: break
        chosen.extend(selected)
        fulltext_selection.update({k:{"year":year,"policy":"recent_facet_round_robin"} for k in selected})
    chosen = chosen[:fulltext_limit]
    fulltext_failures = []
    for index, key in enumerate(chosen):
        if "full_text" not in records[key]["sections"]:
            try:
                records[key] = enrich_fulltext(records[key], archive)
                atomic_json(records_path, records)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                fulltext_failures.append({"paper_id": records[key]["paper_id"], "error": type(exc).__name__})
        progress(f"{domain}: full text {index+1}/{len(chosen)}", flush=True)
    store = LiteratureStore(directory / "literature.sqlite")
    try:
        keys = store.add_many(records.values())
        if len(keys) >= 2:
            seal(directory / ("snapshot-" + digest(sorted(paper_record(p)["content_hash"] for p in records.values()))[:12] + ".json"),
                 store.freeze(keys, cutoff).payload)
    finally:
        store.close()
    normalized = sorted((paper_record(p) for p in records.values()), key=lambda p:p["key"])
    atomic_text(directory / "papers.jsonl", "".join(json.dumps(p, ensure_ascii=False)+"\n" for p in normalized))
    years = Counter(timestamp(p["first_public_at"]).year for p in normalized)
    full_years = Counter(timestamp(p["first_public_at"]).year for p in normalized if "full_text" in p["sections"])
    manifest = {**specification, "generated_at": now(), "data_directory": str(directory),
        "papers": len(normalized), "fulltext_papers": sum(full_years.values()), "by_year": dict(sorted(years.items())),
        "fulltext_by_year": dict(sorted(full_years.items())), "recent_papers": sum(n for y,n in years.items() if y >= end.year-1),
        "strata": sorted(discovery["strata"], key=lambda s:s["year"]),
        "missing_years": [y for y in range(plan["start_year"],end.year+1) if not years[y]],
        "metadata_failures": failures, "fulltext_failures": fulltext_failures,
        "fulltext_selection":{k:fulltext_selection[k] for k in chosen},
        "coverage_claim": "declared-query coverage only; recent work prioritized; not proof of complete field coverage",
        "missing_version_records": sorted(set(discovery["identifiers"]) - set(records)),
        "fulltext_policy": "version-specific archived PDF and text; historical-year minimum followed by recent emphasis; further PDFs fetched for selected papers"}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def load_corpus(path: Path):
    """Read an immutable, hash-checked release. This function never uses network."""
    manifest=unseal(path)
    if manifest.get('artifact_type')!='paperfuture_corpus_release':
        raise ValueError('expected an immutable PaperFuture corpus release manifest')
    export=path.parent/'papers.jsonl'
    raw=export.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['export_sha256']:
        raise ValueError('published corpus export hash mismatch')
    records=[paper_record(json.loads(line)) for line in raw.decode().splitlines() if line.strip()]
    if len(records)!=manifest['papers'] or len({p['key'] for p in records})!=len(records):
        raise ValueError('published corpus paper count or uniqueness mismatch')
    if any(max(timestamp(p['first_public_at']),timestamp(p['version_public_at']))>timestamp(manifest['cutoff']) for p in records):
        raise ValueError('published corpus contains post-cutoff evidence')
    identity=digest({k:manifest[k] for k in ('domain','cutoff','export_sha256')})
    if manifest['release_id']!=identity:
        raise ValueError('published corpus release identity mismatch')
    return manifest,records


def publish_corpus(domain, root, cutoff, records, *, provenance=None):
    """Publish a new immutable release; never overwrite an earlier export."""
    if domain not in PLANS:raise ValueError('unknown corpus domain')
    end=timestamp(cutoff)
    rows=sorted((paper_record(p) for p in records),key=lambda p:p['key'])
    if len(rows)<2 or len({p['key'] for p in rows})!=len(rows):
        raise ValueError('a corpus release needs at least two distinct paper versions')
    if any(max(timestamp(p['first_public_at']),timestamp(p['version_public_at']))>end for p in rows):
        raise ValueError('cannot publish papers later than the corpus cutoff')
    text=''.join(json.dumps(p,ensure_ascii=False,sort_keys=True)+'\n' for p in rows)
    core={'domain':domain,'cutoff':cutoff,'export_sha256':hashlib.sha256(text.encode()).hexdigest()}
    identity=digest(core)
    directory=root/'corpora'/domain/'releases'/identity
    with exclusive(directory/'.lock',blocking=True):
        path=directory/'manifest.json'
        if path.exists():
            existing,_=load_corpus(path)
            return {**existing,'manifest_path':str(path)}
        years=Counter(p['first_public_at'][:4] for p in rows)
        manifest={**core,'artifact_type':'paperfuture_corpus_release','release_id':identity,
            'created_at':now(),'papers':len(rows),'fulltext_papers':sum('full_text' in p['sections'] for p in rows),
            'by_year':dict(sorted(years.items())),
            'recent_papers':sum(n for y,n in years.items() if int(y)>=end.year-1),
            'missing_years':[y for y in range(PLANS[domain]['start_year'],end.year+1) if str(y) not in years],
            'provenance':provenance or {},
            'coverage_claim':'declared-query coverage; independent field coverage review remains necessary'}
        # Manifest is the publication boundary. A failed/in-progress write is
        # never returned as a usable release.
        atomic_text(directory/'papers.jsonl',text)
        seal(path,manifest)
    atomic_json(root/'corpora'/domain/'latest.json',{'release_id':identity,'manifest_path':str(path)})
    return {**manifest,'manifest_path':str(path)}


def publish_source(domain, root, cutoff, source):
    if source.suffix=='.jsonl':
        rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        rows=unseal(source)['papers']
    return publish_corpus(domain,root,cutoff,rows,provenance={
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'operation':'freeze_existing_local_records_without_network'})


def update_corpus(manifest_path, root, cutoff, *, overlap_days=7, fulltext_limit=24, extra_ids=(), replay=False):
    """Fetch only a new date window and requested missing IDs; preserve history.

    An incomplete metadata window never advances the published release. Missing
    PDFs remain explicitly abstract-only. No LLM or embedding API is called.
    """
    base,rows=load_corpus(manifest_path)
    end=timestamp(cutoff);previous=timestamp(base['cutoff'])
    if end<=previous:raise ValueError('corpus update cutoff must advance the previous release')
    if not 0<=overlap_days<=365 or fulltext_limit<0:raise ValueError('invalid update limits')
    domain=base['domain'];papers={p['key']:p for p in rows}
    begin=previous-dt.timedelta(days=overlap_days)
    directory=root/'corpora'/domain/'updates'/digest({'base':base['release_id'],'cutoff':cutoff,
        'overlap_days':overlap_days,'extra_ids':sorted(extra_ids),'fulltext_limit':fulltext_limit})
    archive=Archive(root/'cache/arxiv',replay=replay)
    with exclusive(directory/'.lock',blocking=True):
        found=set()
        def page(raw,path):
            values=parse_arxiv(raw)
            found.update(p['paper_id'][6:]+'v1' for p in values)
            return values
        query=' OR '.join('('+q+')' for name,q in PLANS[domain]['facets'].items() if name!='historical_foundations')
        window=discover_window(archive,query,begin,end,page)
        audit={'base_manifest':str(manifest_path),'base_release':base['release_id'],'cutoff':cutoff,
            'overlap_days':overlap_days,'window':window,'extra_ids':list(extra_ids),'failures':[],
            'new_keys':[],'fulltext_enriched':[],'status':'collecting'}
        atomic_json(directory/'audit.json',audit)
        if window['error'] or window['matched']!=window['retrieved']:
            raise ValueError('incremental discovery incomplete; existing release preserved; inspect update audit')
        for identifier in extra_ids:
            if not re.fullmatch(r'arxiv:\d{4}\.\d{4,5}(?:@v1)?',identifier):
                raise ValueError('supplement IDs must be canonical arxiv IDs with optional @v1')
            found.add(identifier.removeprefix('arxiv:').removesuffix('@v1')+'v1')
        known={p['paper_id'][6:]+p['version_id'] for p in rows if p['paper_id'].startswith('arxiv:')}
        missing=sorted(found-known)
        for offset in range(0,len(missing),50):
            batch=missing[offset:offset+50]
            raw,_=archive.query({'id_list':','.join(batch),'max_results':len(batch)})
            values=parse_arxiv(raw)
            if {p['paper_id'][6:]+p['version_id'] for p in values}!=set(batch):
                raise ValueError('incremental explicit-version metadata mismatch; old release preserved')
            for raw_paper in values:
                p=paper_record(raw_paper)
                if max(timestamp(p['first_public_at']),timestamp(p['version_public_at']))<=end:
                    papers[p['key']]=p;audit['new_keys'].append(p['key'])
            atomic_json(directory/'audit.json',audit)
        # Optional evidence enrichment is confined to this maintenance command.
        priority=sorted(audit['new_keys'],key=lambda k:(papers[k]['first_public_at'],k),reverse=True)
        priority+=sorted(k for k,p in papers.items() if k not in priority and 'full_text' not in p['sections'])
        chosen=[k for k in priority if 'full_text' not in papers[k]['sections'] and papers[k]['paper_id'].startswith('arxiv:')][:fulltext_limit]
        for key in chosen:
            try:
                papers[key]=paper_record(enrich_fulltext(papers[key],archive));audit['fulltext_enriched'].append(key)
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                audit['failures'].append({'key':key,'error':type(exc).__name__,'result':'abstract_only'})
        result=publish_corpus(domain,root,cutoff,papers.values(),provenance={
            'base_release':base['release_id'],'update_audit':str(directory/'audit.json'),
            'new_papers':len(audit['new_keys']),'enriched_fulltexts':len(audit['fulltext_enriched'])})
        audit.update(status='published',manifest_path=result['manifest_path'])
        atomic_json(directory/'audit.json',audit)
        return result
