"""Read-only audit of real dataset artifacts and a completed PaperFuture run.

This verifies storage, provenance and controller consistency, not scientific
novelty or expert agreement. No model or literature network calls are made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperfuture.contracts import SCHEMAS, require_refs, validate
from paperfuture.dataset import data_root
from paperfuture.exploration import (decide, experiences_from_rounds, score_assessment,
    validate_assessment, validate_critique, validate_proposal, validate_simulation)
from paperfuture.graph import validate_graph
from paperfuture.literature import Snapshot, paper_record, round_context
from paperfuture.pipeline import implementation_fingerprint
from paperfuture.storage import atomic_json, now, read_json, timestamp, unseal


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(directory, root, checked_files):
    manifest = read_json(directory / "manifest.json")
    export = directory / "papers.jsonl"
    papers = [json.loads(line) for line in export.read_text().splitlines() if line.strip()]
    keys, years, full_years = set(), Counter(), Counter()
    for paper in papers:
        assert paper == paper_record(paper), f"paper normalization/hash mismatch: {paper['key']}"
        assert paper["key"] not in keys, f"duplicate paper version: {paper['key']}"
        keys.add(paper["key"])
        assert timestamp(paper["version_public_at"]) <= timestamp(manifest["cutoff"])
        assert paper["version_id"] == "v1" and paper["source_url"].endswith("v1")
        year = str(timestamp(paper["first_public_at"]).year)
        years[year] += 1
        if "full_text" in paper["sections"]:
            full_years[year] += 1
            artifacts = paper["artifacts"]
            for kind in ("pdf", "text"):
                path = Path(artifacts[kind]).resolve()
                assert path.is_relative_to(root), f"paper artifact outside data root: {path}"
                if path not in checked_files:
                    checked_files[path] = file_hash(path)
                assert checked_files[path] == artifacts[kind+"_sha256"], f"file hash mismatch: {path}"
            receipt = read_json(Path(artifacts["pdf"]+".json"))
            assert receipt["sha256"] == artifacts["pdf_sha256"]
            assert receipt["url"].endswith(paper["paper_id"][6:]+"v1")
            assert hashlib.sha256(paper["sections"]["full_text"].encode()).hexdigest() == artifacts["text_sha256"]
    assert len(keys) == manifest["papers"] and dict(years) == manifest["by_year"]
    assert dict(full_years) == manifest["fulltext_by_year"]
    recent = sum(n for year, n in years.items() if int(year) >= timestamp(manifest["cutoff"]).year-1)
    assert recent == manifest["recent_papers"]
    gaps = [s["year"] for s in manifest["strata"] if s["status"] != "complete" or s["truncated"]]
    return {"domain": manifest["domain"], "directory": str(directory), "papers": len(keys),
        "fulltext_papers": sum(full_years.values()), "by_year": dict(sorted(years.items())),
        "fulltext_by_year": dict(sorted(full_years.items())), "recent_papers": recent,
        "recent_fraction": round(recent/len(keys), 6) if keys else None,
        "source_export_sha256": file_hash(export), "incomplete_query_years": gaps,
        "missing_years": manifest["missing_years"],
        "missing_version_records": len(manifest["missing_version_records"]),
        "metadata_failures": len(manifest["metadata_failures"]),
        "fulltext_failures": len(manifest["fulltext_failures"]),
        "storage_hash_and_version_checks": "passed", "coverage_claim": manifest["coverage_claim"]}


def audit_run(directory):
    spec, state = unseal(directory/"spec.json"), unseal(directory/"state.json")
    assert spec["implementation_fingerprint"] == implementation_fingerprint(), "use the matching source revision for auditing this run"
    snapshot = Snapshot(unseal(directory/"snapshot.json"))
    selected, graph = unseal(directory/"selected.json"), unseal(directory/"trajectory.json")
    validate_graph({k:graph[k] for k in ("stages", "edges", "coverage_gaps")}, snapshot, selected)
    if graph.get("stage_reconciliation"):
        reconciliation=graph["stage_reconciliation"]
        validate(reconciliation,SCHEMAS["graph_merge"])
        original={n["id"]:n for n in graph["batch_stages"]}
        used=[i for g in reconciliation["groups"] for i in g["member_stage_ids"]]
        assert len(used)==len(original) and set(used)==set(original)
        for group,node in zip(reconciliation["groups"],graph["stages"]):
            representative=original[group["representative_stage_id"]]
            assert node["mechanism"]==representative["mechanism"]
            assert set(node["paper_keys"])=={k for i in group["member_stage_ids"] for k in original[i]["paper_keys"]}
            assert all(c[k] for c in group["comparisons"] for k in ("same_problem","same_mechanism","same_assumptions","same_capability_boundary"))
    for key, extraction in selected["extractions"].items():
        validate(extraction, SCHEMAS["extract"])
        require_refs(extraction["evidence_ids"], snapshot.spans(key), "selected extraction", True)
    accepted = {review["edge_id"] for review in graph["edge_reviews"] if review["verdict"] == "supported"}
    assert {edge["id"] for edge in graph["edges"]} == accepted
    base = unseal(directory/"graph_context.json")
    completed, model_ids = [], set()
    for slot in state["rounds"]:
        rd = directory/"rounds"/f"{slot['number']:02}"
        row, context = unseal(rd/"evaluated.json"), unseal(rd/"evidence_context.json")
        assert unseal(rd/"experience_memory.json") == experiences_from_rounds(completed)
        assert context == round_context(base, snapshot, spec, slot["input_feedback"], experiences_from_rounds(completed))
        validate(row["proposal"], SCHEMAS["propose"])
        inherited = list((slot["input_feedback"] or {}).get("inherited_assumptions", []))
        used = {item["experience_id"] for item in row["proposal"]["learning_trace"]}
        inherited += [a for e in experiences_from_rounds(completed) if e["id"] in used for a in e["assumptions"]]
        validate_proposal(row["proposal"], context, inherited)
        for member in row["simulation"]["members"]:
            model_ids.add(member["model_id"])
            validate(member["result"], SCHEMAS["simulate"])
            validate_simulation(member["result"], row["proposal"], context)
        for member in row["critique"]["members"]:
            model_ids.add(member["model_id"])
            validate(member["result"], SCHEMAS["critique"])
            validate_critique(member["result"], row["proposal"], context)
        for member in row["discussion"]["members"]:
            validate(member["result"], SCHEMAS["discuss"])
            for prediction in member["result"]["revised_predictions"]:
                require_refs(prediction["evidence_ids"], context["evidence"], "discussion evidence")
                require_refs(prediction["assumption_ids"], {a["id"] for a in row["proposal"]["assumptions"]}, "discussion assumptions")
        validate(row["assessment"], SCHEMAS["assess"])
        validate_assessment(row["assessment"], row["critique"], context, completed+[row], experiences_from_rounds(completed+[row]))
        assert score_assessment(row["assessment"], row["critique"], spec["policy"]) == row["scores"]
        assert decide(completed+[row], spec) == unseal(rd/"decision.json") == slot["decision"]
        completed.append(row)
    final = read_json(directory/"final_idea.json")
    assert state["status"] == final["status"] == "complete"
    assert len(completed) == final["completed_rounds"] == final["reserved_rounds"] <= spec["max_rounds"] <= 4
    assert final["candidate"] == completed[final["selected_round"]-1]
    return {"run_id": final["run_id"], "snapshot_papers": len(snapshot.papers),
        "selected_papers": len(selected["selected_keys"]), "selection_years": selected["year_counts"],
        "stages": len(graph["stages"]), "proposed_edges": len(graph["proposed_edges"]),
        "supported_edges": len(graph["edges"]), "edge_weights": [e["weight"] for e in graph["edges"]],
        "completed_rounds": len(completed), "stop_reason": final["stop_reason"],
        "recommendation": final["recommendation"], "selected_round": final["selected_round"],
        "world_model_ids": sorted(model_ids), "usage": final["usage"],
        "schema_provenance_temporal_graph_memory_and_controller_checks": "passed",
        "human_semantic_review": "pending", "scientific_quality": "not_established_by_this_audit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = (args.data_root or data_root()).resolve()
    files = {}
    report = {"audited_at": now(), "scope": "software_and_data_integrity_not_scientific_evaluation",
        "implementation_fingerprint": implementation_fingerprint(),
        "datasets": [audit_dataset(p.parent, root, files) for p in sorted((root/"datasets").glob("*/*/manifest.json"))],
        "verified_source_files": len(files)}
    if args.run:
        report["run"] = audit_run(args.run)
    atomic_json(args.output, report)
    summary={"datasets": [{k:d[k] for k in ("domain", "papers", "fulltext_papers", "incomplete_query_years")} for d in report["datasets"]],
             "output": str(args.output)}
    if "run" in report:
        summary["run"]={k:v for k,v in report["run"].items() if k!="usage"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
