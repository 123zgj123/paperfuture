"""Compare the index server's ranking modes on real agent queries, judged blind by a model.

    python3.11 scripts/literature/eval_retrieval.py --server http://127.0.0.1:8765 \
        --traces <AgentIdeaBench>/release_data/derived/active_traces.csv.gz --out /data/retrieval-eval.json \
        --judge-model qwen3.6-plus --queries 40

Queries are the SEARCH strings that Active-track agents actually issued on AgentIdeaBench's
eight densely scored CS subfields (sampled with a fixed seed). For each query every mode's top
10 are pooled, shuffled, and shown to the judge without saying which mode found what; the judge
grades each paper 2 (about the query's topic), 1 (related) or 0 (not relevant). Reported per
mode: P@10 (grade >= 1), nDCG@10 (gains 0/1/2, ideal from the pooled grades) and latency.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path

CS_DENSE = {
    "3D scene understanding neural radiance field", "adversarial robustness machine learning defense",
    "continual learning catastrophic forgetting", "federated learning privacy distributed training",
    "large language model reasoning chain of thought", "neural architecture search efficient model",
    "self-supervised representation learning contrastive", "time series forecasting deep learning"}
MODES = ("bm25", "bi", "cross")
JUDGE = ("You grade literature search results. A researcher typed the query below into a paper search "
         "engine. For each numbered paper (title and abstract), grade 2 if it is about what the query "
         "asks for, 1 if it is related but not what the query asks for, 0 if it is not relevant. Judge "
         "relevance only, not quality or recency. Reply with JSON only: {\"grades\": {\"<number>\": 0|1|2}}")


def queries(path: Path, n: int, seed: int) -> list[str]:
    found = []
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            if row["subdomain"] in CS_DENSE and row["tool"] == "search_papers" and row["query"].strip():
                found.append(" ".join(row["query"].split()))
    unique = sorted(set(found))
    random.Random(seed).shuffle(unique)
    return unique[:n]


def search(server: str, q: str, mode: str, before: str) -> tuple[list[dict], float]:
    started = time.monotonic()
    url = f"{server}/search?{urllib.parse.urlencode({'q': q, 'limit': 10, 'before': before, 'mode': mode})}"
    with urllib.request.urlopen(url, timeout=300) as resp:
        rows = json.load(resp)
    return rows, time.monotonic() - started


def judge(query: str, papers: list[dict], *, base_url: str, key: str, model: str) -> dict[str, int]:
    listing = "\n\n".join(f"[{i}] {p['title']}\n{(p.get('abstract') or '')[:700]}" for i, p in enumerate(papers, 1))
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 1500,
                       "messages": [{"role": "system", "content": JUDGE},
                                    {"role": "user", "content": f"Query: {query}\n\n{listing}"}]}).encode()
    for attempt in range(4):
        try:
            request = urllib.request.Request(f"{base_url}/chat/completions", data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as resp:
                text = json.load(resp)["choices"][0]["message"]["content"]
            grades = json.loads(re.search(r"\{.*\}", text, re.S).group(0))["grades"]
            return {str(k): int(v) for k, v in grades.items()}
        except Exception as exc:
            print(f"judge retry {attempt + 1}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


def ndcg(gains: list[int], ideal: list[int]) -> float:
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:10]))
    best = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(ideal, reverse=True)[:10]))
    return dcg / best if best else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--before", default="2026-05-31")
    parser.add_argument("--judge-model", default="qwen3.6-plus")
    args = parser.parse_args()
    base_url = os.environ["JUDGE_BASE_URL"].rstrip("/")
    key = Path(os.environ["JUDGE_KEY_FILE"]).read_text().strip()
    rows = []
    for number, q in enumerate(queries(args.traces, args.queries, args.seed), 1):
        results, latency = {}, {}
        for mode in MODES:
            results[mode], latency[mode] = search(args.server, q, mode, args.before)
        pooled = {}
        for mode in MODES:
            for paper in results[mode]:
                pooled.setdefault(paper["id"], paper)
        order = list(pooled)
        random.Random(f"{args.seed}:{q}").shuffle(order)
        grades = judge(q, [pooled[i] for i in order], base_url=base_url, key=key, model=args.judge_model)
        grade_of = {pid: grades.get(str(n)) for n, pid in enumerate(order, 1)}
        if any(g is None for g in grade_of.values()):
            print(f"skip (incomplete grades): {q}", flush=True)
            continue
        ideal = list(grade_of.values())
        row = {"query": q, "pooled": len(order), "modes": {}}
        for mode in MODES:
            gains = [grade_of[p["id"]] for p in results[mode]]
            row["modes"][mode] = {"p10": sum(g >= 1 for g in gains) / 10, "p10_strict": sum(g == 2 for g in gains) / 10,
                                  "ndcg10": round(ndcg(gains, ideal), 4), "seconds": round(latency[mode], 3),
                                  "top": [p["id"] for p in results[mode]]}
        rows.append(row)
        print(json.dumps({"n": number, "q": q[:60], **{m: row["modes"][m]["ndcg10"] for m in MODES}}), flush=True)
    summary = {mode: {metric: round(statistics.mean(r["modes"][mode][metric] for r in rows), 4)
                      for metric in ("p10", "p10_strict", "ndcg10", "seconds")} for mode in MODES}
    args.out.write_text(json.dumps({"judge": args.judge_model, "queries": len(rows), "before": args.before,
                                    "summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
