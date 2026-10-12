"""Run the real pipeline with explicitly synthetic HTTP fixtures, without API spend.

This verifies software behavior; its papers, responses and ideas are NOT research
evidence. The production CLI never imports or falls back to this fixture server.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from paperfuture.pipeline import run
from paperfuture_support import SyntheticEndpoint, configuration, papers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "var/paperfuture/offline-demo")
    parser.add_argument("--four-rounds", action="store_true", help="exercise the global four-round limit instead of early stopping")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    source = args.workspace.resolve() / "SYNTHETIC-papers.jsonl"
    source.write_text("\n".join(json.dumps(p) for p in papers()) + "\n")
    config = configuration(source, run_id="SYNTHETIC-four-rounds" if args.four_rounds else "SYNTHETIC-early-stop")
    config["question"] = "OFFLINE SOFTWARE DEMO ONLY: test the PaperFuture six-step pipeline with synthetic agent-security fixtures."
    endpoint = SyntheticEndpoint(high_value=not args.four_rounds)
    with patch.dict(os.environ, {"PAPERFUTURE_TEST_KEY": "synthetic-not-a-real-key"}), patch("paperfuture.api.urllib.request.urlopen", endpoint):
        result = run(config, args.workspace, progress=print)
    print("SYNTHETIC SOFTWARE DEMO — no real papers, model inference, or scientific findings.")
    print(json.dumps({k: result[k] for k in ("run_id", "status", "completed_rounds", "stop_reason")}, indent=2))
    print(args.workspace.resolve() / "runs" / config["run_id"] / "IDEA.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
