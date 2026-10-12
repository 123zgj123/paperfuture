"""Serve the local arXiv index over HTTP on this machine, with a second-stage reranker.

    <retrieval-venv>/bin/python scripts/literature/serve_index.py --db /data/literature/arxiv.sqlite \
        --port 8765 [--mode cross]

Retrieval is two-stage. BM25 over title and abstract (SQLite FTS5) recalls the top ``--pool``
candidates; a neural model then reorders them: ``bi`` embeds query and candidates with
bge-small-en-v1.5 and ranks by cosine, ``cross`` scores each (query, candidate) pair with
bge-reranker-base, ``bm25`` keeps the first stage's order. Candidates are embedded at query
time, so nothing is precomputed or stored for the whole collection.

Endpoints (JSON): ``/search?q=&limit=&before=&after=&mode=``, ``/related?id=&limit=&before=``,
``/paper?id=``, ``/health``. Every system that searches literature in an experiment goes through
the same server, so they all see the same ranking. Answers are cached in memory.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from farfield.services.localindex import LocalIndex  # noqa: E402

MODES = ("bm25", "bi", "cross")


class Ranker:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.lock = threading.Lock()
        self.bi = self.cross = None
        if mode == "bi":
            from sentence_transformers import SentenceTransformer
            self.bi = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        elif mode == "cross":
            from sentence_transformers import CrossEncoder
            self.cross = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=384)

    @staticmethod
    def text(row: dict) -> str:
        return f"{row['title']}. {(row.get('abstract') or '')[:1200]}"

    def rank(self, query: str, rows: list[dict]) -> list[dict]:
        if self.mode == "bm25" or len(rows) < 2:
            return rows
        with self.lock:
            if self.mode == "bi":
                import numpy as np
                vectors = self.bi.encode(["Represent this sentence for searching relevant passages: " + query]
                                         + [self.text(r) for r in rows], normalize_embeddings=True,
                                         batch_size=32)
                scores = (vectors[1:] @ vectors[0]).tolist()
            else:
                scores = self.cross.predict([(query, self.text(r)) for r in rows], batch_size=32).tolist()
        order = sorted(range(len(rows)), key=lambda i: -scores[i])
        return [{**rows[i], "score": round(float(scores[i]), 4)} for i in order]


def make_handler(index: LocalIndex, rankers: dict[str, Ranker], default: str, pool: int):
    @functools.lru_cache(maxsize=20000)
    def search(q: str, limit: int, before: str, after: str, mode: str) -> str:
        rows = index.search(q, limit=max(pool, limit), before=before, after=after)
        return json.dumps(rankers[mode].rank(q, rows)[:limit])

    @functools.lru_cache(maxsize=5000)
    def related(pid: str, limit: int, before: str, mode: str) -> str:
        paper = index.get(pid)
        if paper is None:
            return "[]"
        cutoff = min(filter(None, (paper["published"], before))) if before else paper["published"]
        query = f"{paper['title']} {(paper['abstract'] or '')[:400]}"
        rows = index.search(query, limit=max(pool, limit), before=cutoff, exclude=[paper["id"]])
        return json.dumps(rankers[mode].rank(paper["title"], rows)[:limit])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, body: str, code: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            mode = q.get("mode") or default
            if mode not in rankers:
                return self._send(json.dumps({"error": f"mode {mode} not loaded"}), 400)
            try:
                if url.path == "/search":
                    return self._send(search(q.get("q", ""), int(q.get("limit") or 10), q.get("before", ""),
                                             q.get("after", ""), mode))
                if url.path == "/related":
                    return self._send(related(q.get("id", ""), int(q.get("limit") or 15), q.get("before", ""), mode))
                if url.path == "/paper":
                    return self._send(json.dumps(index.get(q.get("id", ""))))
                if url.path == "/health":
                    return self._send(json.dumps({"papers": index.count(), "modes": list(rankers),
                                                  "default": default}))
            except Exception as exc:  # a bad query must not take the server down
                return self._send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), 500)
            self._send(json.dumps({"error": "unknown endpoint"}), 404)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=MODES, default="cross", help="default ranking")
    parser.add_argument("--load", nargs="*", choices=MODES, default=None, help="modes to load (default: all)")
    parser.add_argument("--pool", type=int, default=100, help="BM25 candidates handed to the reranker")
    args = parser.parse_args()
    started = time.monotonic()
    rankers = {mode: Ranker(mode) for mode in (args.load or MODES)}
    index = LocalIndex(args.db)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(index, rankers, args.mode, args.pool))
    print(json.dumps({"listening": args.port, "papers": index.count(), "modes": list(rankers),
                      "load_seconds": round(time.monotonic() - started, 1)}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
