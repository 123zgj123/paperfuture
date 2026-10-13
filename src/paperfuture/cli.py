from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import DOMAINS, credential, load_config
from .contracts import SCHEMAS
from .pipeline import run
from .storage import atomic_json, read_json, unseal
from .dataset import PLANS, data_root, harvest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="paperfuture", description="Versioned literature → weighted trajectory → leap → conditional world model → at most four rounds → idea")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "prepare", "doctor"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "doctor":
            command.add_argument("--workspace", type=Path, help="data-volume workspace; defaults to PAPERFUTURE_DATA_ROOT")
            command.add_argument("--replay", action="store_true", help="refuse network calls; recover from local recordings only")
    status = commands.add_parser("status")
    status.add_argument("run_directory", type=Path)
    schemas = commands.add_parser("schemas")
    schemas.add_argument("--output", type=Path, required=True)
    commands.add_parser("domains")
    collection=commands.add_parser("collect",help="collect versioned literature into a data volume without LLM calls")
    collection.add_argument("--data-root",type=Path)
    collection.add_argument("--domain",choices=["all",*PLANS],default="all")
    collection.add_argument("--cutoff",required=True)
    collection.add_argument("--fulltext-limit",type=int,default=100)
    collection.add_argument("--replay",action="store_true")
    bench = commands.add_parser("bench")
    bench.add_argument("--cases", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    ratings=commands.add_parser("bench-graph",help="evaluate heldout expert relation/score labels without model calls")
    ratings.add_argument("--run",type=Path,required=True)
    ratings.add_argument("--annotations",type=Path,required=True)
    ratings.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            for domain in PLANS if args.domain=="all" else [args.domain]:
                result=harvest(domain,args.data_root or data_root(),args.cutoff,fulltext_limit=args.fulltext_limit,replay=args.replay)
                print(json.dumps({k:result[k] for k in ("domain","papers","fulltext_papers","by_year","data_directory")},ensure_ascii=False))
        elif args.command == "domains":
            print(json.dumps(DOMAINS, ensure_ascii=False, indent=2))
        elif args.command == "schemas":
            atomic_json(args.output, SCHEMAS)
        elif args.command == "status":
            state = unseal(args.run_directory / "state.json")
            print(json.dumps({k: state[k] for k in ("status", "phase", "rounds", "stop_reason")}, ensure_ascii=False, indent=2))
        elif args.command == "bench-graph":
            from .bench import audit_graph_ratings
            atomic_json(args.output,audit_graph_ratings(args.run,read_json(args.annotations)))
            print(f"Heldout graph judgment audit: {args.output}")
        elif args.command == "bench":
            from .bench import audit_cases
            atomic_json(args.output, audit_cases(read_json(args.cases), args.cases.resolve().parent))
            print(f"Benchmark chronology audit: {args.output}")
        else:
            config = load_config(args.config)
            if args.command == "doctor":
                # Read-only local preflight. Model-list access is not a proof of model identity.
                checks = []
                for role, model in config["models"].items():
                    credential(model)
                    checks.append({"role": role, "model": model["model"], "base_url": model["base_url"], "credential_available": True})
                for source in config["sources"]:
                    if source["type"] == "jsonl" and not Path(source["path"]).is_file():
                        raise ValueError(f"missing literature file: {source['path']}")
                print(json.dumps({"ready": True, "models": checks, "network_tested": False}, indent=2))
            else:
                args.workspace=args.workspace or data_root()
                if args.replay:
                    config["api_mode"] = "replay"
                result = run(config, args.workspace, prepare_only=args.command == "prepare", progress=lambda s: print(s, file=sys.stderr, flush=True))
                print(json.dumps({k: result.get(k) for k in ("run_id", "status", "recommendation", "completed_rounds", "stop_reason")}, ensure_ascii=False, indent=2))
                print(f"Artifacts: {args.workspace.resolve() / 'runs' / config['run_id']}")
    except KeyboardInterrupt:
        print("PaperFuture interrupted; rerun the same configuration to resume persisted work.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"PaperFuture failed: {exc}", file=sys.stderr)
        return 1
    return 0
