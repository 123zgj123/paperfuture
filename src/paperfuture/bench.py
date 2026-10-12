"""Offline chronology auditing and human-annotation metrics for PaperFutureBench.

Future labels are consumed only here, after generation, never by the pipeline.
This is evaluation infrastructure, not a released/validated benchmark dataset.
"""
from __future__ import annotations

from pathlib import Path
import statistics

from .storage import read_json, timestamp, unseal


def audit_cases(cases: list[dict], base: Path) -> dict:
    results, ids = [], set()
    for case in cases:
        if case["case_id"] in ids:
            raise ValueError("duplicate benchmark case IDs")
        ids.add(case["case_id"])
        directory = (base / case["run_directory"]).resolve()
        spec = unseal(directory / "spec.json")
        snapshot = unseal(directory / "snapshot.json")
        final = read_json(directory / "final_idea.json")
        cutoff = timestamp(spec["cutoff"])
        if snapshot["cutoff"] != spec["cutoff"] or final["cutoff"] != spec["cutoff"]:
            raise ValueError("benchmark run cutoff mismatch")
        if any(timestamp(p["version_public_at"]) > cutoff or timestamp(p["first_public_at"]) > cutoff for p in snapshot["papers"]):
            raise ValueError("future content is present in generation snapshot")
        label_ids = set()
        for label in case["future_papers"]:
            if timestamp(label["first_public_at"]) <= cutoff:
                raise ValueError("future labels must first become public AFTER cutoff, regardless of acceptance date")
            if label["paper_id"] in label_ids:
                raise ValueError("duplicate future labels")
            label_ids.add(label["paper_id"])
        if label_ids & {p["paper_id"] for p in snapshot["papers"]}:
            raise ValueError("future label paper also appears in the generation snapshot")
        judgments = case.get("human_judgments", [])
        # A lexical match is deliberately not used as a surrogate for future value.
        for judgment in judgments:
            if judgment["future_paper_id"] not in label_ids:
                raise ValueError("judgment refers to an unknown future label")
            if type(judgment["mechanism_match"]) is not int or not 0 <= judgment["mechanism_match"] <= 4:
                raise ValueError("mechanism_match must be an ordinal integer 0..4")
            for dimension in ("novelty", "importance", "feasibility"):
                score = judgment.get(dimension)
                if score is not None and (type(score) is not int or not 0 <= score <= 4):
                    raise ValueError(f"human {dimension} must be null or an integer 0..4")
            if not judgment.get("rater_id") or not judgment.get("rationale"):
                raise ValueError("human judgments require rater identity and a rationale")
        grouped = {}
        seen = set()
        for judgment in judgments:
            pair = (judgment["future_paper_id"], judgment["rater_id"])
            if pair in seen:
                raise ValueError("duplicate judgment from one rater on one future paper")
            seen.add(pair)
            grouped.setdefault(judgment["future_paper_id"], []).append(judgment["mechanism_match"])
        matches = {k: statistics.mean(v) for k,v in grouped.items()}
        results.append({"case_id": case["case_id"], "input_chronology_valid": True,
            "model_leakage_status": "unverified", "model_vintage_evidence": spec["model_vintage_evidence"],
            "run_status": final["status"], "selected_round": final["selected_round"],
            "future_label_count": len(label_ids), "judged_label_count": len(grouped),
            "human_mechanism_match_by_label": matches,
            "has_mean_match_at_least_3": any(s >= 3 for s in matches.values()) if matches else None,
            "usage": {k: final["usage"][k] for k in ("network_attempts", "reported_tokens", "unknown_billing_attempts")}})
    judged = [r for r in results if r["has_mean_match_at_least_3"] is not None]
    return {"cases": results, "case_count": len(results), "judged_case_count": len(judged),
        "human_future_match_rate": statistics.mean(int(r["has_mean_match_at_least_3"]) for r in judged) if judged else None,
        "limitations": ["Chronology is audited from declared source-version metadata; source attestations still require review.",
                        "API model identity/training cutoff is not proven by its name or release date.",
                        "A future-paper mechanism match is one outcome; unmatched ideas can still be valuable.",
                        "Use independently blinded raters, predeclared splits, equal generation budgets and report uncertainty before claiming superiority."]}
