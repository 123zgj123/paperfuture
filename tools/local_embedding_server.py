"""Loopback-only embedding endpoint for an audited local SentenceTransformer.

Run this in the separate data-volume CPU environment. PaperFuture's LLM roles
continue to use their configured APIs. Inputs must have explicit query/document
prefixes; long input is rejected rather than silently truncated.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import time


def validate_inputs(payload, model_id):
    if not isinstance(payload, dict) or set(payload) - {"model", "input", "encoding_format"}:
        raise ValueError("expected model, input and optional encoding_format")
    if payload.get("model") != model_id:
        raise ValueError("model ID must match the fixed local inference profile")
    if payload.get("encoding_format", "float") != "float":
        raise ValueError("only float embeddings are supported")
    texts = payload.get("input")
    if not isinstance(texts, list) or not 1 <= len(texts) <= 32:
        raise ValueError("input must contain between 1 and 32 strings")
    if any(not isinstance(t, str) or not t.startswith(("query: ", "passage: "))
           or not t.split(": ", 1)[1].strip() for t in texts):
        raise ValueError("each input requires a query: or passage: prefix and nonempty text")
    if sum(len(t.encode("utf-8")) for t in texts) > 512_000:
        raise ValueError("input exceeds local request byte limit")
    return texts


def verify_checkpoint(directory):
    manifest = json.loads((directory / "download-manifest.json").read_text())
    for item in manifest["files"]:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or path.stat().st_size != item["bytes"]:
            raise ValueError("invalid local model artifact")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                h.update(chunk)
        if h.hexdigest() != item["sha256"] or not item["upstream_hash_verified"]:
            raise ValueError("model artifact failed upstream hash verification")
    return manifest


class Encoder:
    def __init__(self, directory, threads, max_tokens):
        # Set these before importing HF libraries: loading cannot fetch anything.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        manifest = verify_checkpoint(directory)
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.model = SentenceTransformer(str(directory), device="cpu", local_files_only=True,
            trust_remote_code=False, model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
        self.model.max_seq_length = max_tokens
        self.max_tokens = max_tokens
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.profile = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
            "checkpoint_manifest_sha256": hashlib.sha256((directory / "download-manifest.json").read_bytes()).hexdigest(),
            "server_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "device": "cpu", "dtype": "bfloat16", "attention": "sdpa", "batch_size": 1,
            "threads": threads, "max_tokens": max_tokens, "dimensions": self.dimension,
            "normalize": True, "long_input_policy": "reject", "query_prefix": "query: ", "document_prefix": "passage: ",
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "sentence-transformers", "tokenizers", "numpy")}}
        identity = hashlib.sha256(json.dumps(self.profile, sort_keys=True).encode()).hexdigest()[:16]
        self.model_id = "nemotron-3-embed-1b-" + manifest["revision"][:12] + "-" + identity

    def encode(self, texts):
        encoded = self.model.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        if max(lengths) > self.max_tokens:
            raise ValueError(f"input has {max(lengths)} tokens; local profile limit is {self.max_tokens}; chunk it explicitly")
        vectors = self.model.encode(texts, batch_size=1, prompt="", normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True).astype("float32").tolist()
        if len(vectors) != len(texts) or any(len(v) != self.dimension or
                not all(math.isfinite(x) for x in v) or abs(sum(x*x for x in v)-1) > .02 for v in vectors):
            raise RuntimeError("invalid local embedding output")
        return vectors, sum(lengths)


def serve(encoder, key, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log request text or credentials.

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key):
                self.reply(401, {"error": "local API credential required"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": encoder.model_id, "object": "model", "owned_by": "local"}]})
            elif self.path == "/health":
                self.reply(200, {"ready": True, "model": encoder.model_id, "profile": encoder.profile})
            else:
                self.reply(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/embeddings":
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 600_000:
                    raise ValueError("invalid request size")
                texts = validate_inputs(json.loads(self.rfile.read(length)), encoder.model_id)
                vectors, tokens = encoder.encode(texts)
            except (ValueError, TypeError, KeyError):
                self.reply(422, {"error": "invalid model, input prefixes, encoding, or token limit; no input was truncated"})
                return
            except Exception:
                self.reply(500, {"error": "local inference failed"})
                return
            self.reply(200, {"object": "list", "model": encoder.model_id,
                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})

    # Binding is deliberately fixed; this is a per-user local dependency.
    with HTTPServer(("127.0.0.1", port), Handler) as server:
        server.timeout = 30
        print(json.dumps({"ready": True, "port": port, "model": encoder.model_id}), flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--profile-file", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or not 1 <= args.max_tokens <= 32768 or not 1024 <= args.port <= 65535:
        parser.error("invalid CPU thread, token or port limit")
    key = args.key_file.read_text().strip()
    if len(key) < 24:
        parser.error("local API key file must contain at least 24 characters")
    started = time.monotonic()
    encoder = Encoder(args.model_directory, args.threads, args.max_tokens)
    args.profile_file.parent.mkdir(parents=True, exist_ok=True)
    args.profile_file.write_text(json.dumps({"model_id": encoder.model_id, "base_url": f"http://127.0.0.1:{args.port}/v1",
        "profile": encoder.profile, "load_seconds": time.monotonic()-started}, indent=2) + "\n")
    serve(encoder, key, args.port)


if __name__ == "__main__":
    main()
