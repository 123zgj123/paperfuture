"""Acquire real, versioned papers into the specified data volume, without LLM calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paperfuture.dataset import PLANS, data_root, harvest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--domain", choices=["all", *PLANS], default="all")
    parser.add_argument("--historical-limit", type=int, default=400)
    parser.add_argument("--recent-limit", type=int, default=1600)
    parser.add_argument("--fulltext-limit", type=int, default=100)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    root = (args.data_root or data_root()).resolve()
    if min(args.historical_limit,args.recent_limit) < 1 or args.fulltext_limit < 0:
        parser.error("invalid acquisition limits")
    for domain in PLANS if args.domain == "all" else [args.domain]:
        result = harvest(domain, root, args.cutoff, historical_limit=args.historical_limit,
                         recent_limit=args.recent_limit, fulltext_limit=args.fulltext_limit, replay=args.replay)
        print(json.dumps({k:result[k] for k in ("domain","papers","fulltext_papers","by_year","data_directory")},ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
