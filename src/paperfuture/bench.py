"""Offline chronology auditing and human-annotation metrics for PaperFutureBench.

Future labels are consumed only here, after generation, never by the pipeline.
This is evaluation infrastructure, not a released/validated benchmark dataset.
"""
from __future__ import annotations

from pathlib import Path
import statistics

from .storage import read_json, timestamp, unseal
from .contracts import DIMENSIONS, EDGE


def _ranks(values):
    return [1+sum(x<v for x in values)+(values.count(v)-1)/2 for v in values]


def audit_graph_ratings(directory: Path, annotations: list[dict]) -> dict:
    """Held-out relation and ordinal-score evaluation; never sends labels to a model.

    Re-run on independently generated graphs/replicates for stability. Expert
    identity is attested by the annotation supplier, not invented by software.
    """
    graph=unseal(directory/"trajectory.json");spec=unseal(directory/"spec.json")
    snapshot=unseal(directory/"snapshot.json")
    papers={p["key"]:p for p in snapshot["papers"]}
    nodes={n["id"]:n for n in graph["stages"]}
    anchors=graph.get("calibration_anchors",spec.get("rating_anchors",[]))
    anchor_pairs={(a["source_key"],a["target_key"]) for a in anchors}
    anchor_ids={a["anchor_id"] for a in anchors}
    membership={k:n["id"] for n in nodes.values() for k in n["paper_keys"]}
    edges={(e["source"],e["target"]):e for e in graph["edges"]}
    examined={(p["source"],p["target"]) for p in graph.get("candidate_pair_audit",{}).get("pairs",[])}
    grouped={};seen=set()
    for a in annotations:
        if a.get("split")!="heldout" or a.get("case_id") in anchor_ids or (a.get("source_key"),a.get("target_key")) in anchor_pairs:
            raise ValueError("heldout graph labels must be disjoint from generation/calibration anchors by ID AND paper pair")
        if not a.get("annotator") or not a.get("rationale"):raise ValueError("expert labels require annotator and rationale")
        key=(a["source_key"],a["target_key"])
        if key[0] not in papers or key[1] not in papers:raise ValueError("annotation endpoint outside frozen snapshot")
        if timestamp(papers[key[0]]["version_public_at"])>=timestamp(papers[key[1]]["version_public_at"]):
            raise ValueError("annotation requires strictly forward source versions")
        identity=(key,a["annotator"])
        if identity in seen:raise ValueError("duplicate expert judgment for one pair")
        seen.add(identity)
        if type(a.get("supported")) is not bool:raise ValueError("supported must be a human Boolean relation judgment")
        weight=a.get("weight")
        if weight is not None and (type(weight) is not int or not 0<=weight<=100):raise ValueError("expert weight must be null or integer 0..100")
        if not a["supported"] and weight is not None:raise ValueError("unsupported relation cannot have a technical change score")
        if a.get("relation") is not None and a["relation"] not in EDGE["properties"]["relation"]["enum"]:
            raise ValueError("unknown annotated development relation")
        if a.get("delta") is not None and (set(a["delta"])!=set(DIMENSIONS) or any(v is not None and (type(v) is not int or not 0<=v<=10) for v in a["delta"].values())):
            raise ValueError("expert dimension ratings must contain the five null or integer 0..10 judgments")
        grouped.setdefault(key,[]).append(a)
    cases=[];predicted=[];gold=[];tp=fp=fn=tn=0;relation_matches=[];dimension_errors={k:[] for k in DIMENSIONS}
    for keys,judgments in sorted(grouped.items()):
        support_values={a["supported"] for a in judgments}
        pair=(membership.get(keys[0]),membership.get(keys[1]));edge=edges.get(pair)
        agreed=len(support_values)==1
        expected=next(iter(support_values)) if agreed else None
        if expected is not None:
            tp+=int(bool(edge) and expected);fp+=int(bool(edge) and not expected)
            fn+=int(not edge and expected);tn+=int(not edge and not expected)
        weights=[a["weight"] for a in judgments if a.get("weight") is not None and a["supported"]]
        mean=statistics.mean(weights) if weights and expected is True else None
        score=edge["weight"] if edge else None
        if mean is not None and score is not None:predicted.append(score);gold.append(mean)
        relations={a["relation"] for a in judgments if a.get("relation") is not None}
        if expected is True and edge and len(relations)==1:relation_matches.append(edge["relation"] in relations)
        if expected is True and edge:
            for dimension in DIMENSIONS:
                values=[a["delta"][dimension] for a in judgments if (a.get("delta") or {}).get(dimension) is not None]
                if values and edge["delta"][dimension] is not None:
                    dimension_errors[dimension].append(abs(statistics.mean(values)-edge["delta"][dimension]))
        cases.append({"source_key":keys[0],"target_key":keys[1],"annotators":[a["annotator"] for a in judgments],
            "expert_support_consensus":expected,"expert_disagreement":not agreed,"expert_weight_mean":mean,
            "expert_weight_range":[min(weights),max(weights)] if weights else None,
            "model_edge_id":edge["id"] if edge else None,"model_weight":score,
            "coverage_status":"selected_same_stage" if pair[0] is not None and pair[0]==pair[1] else
                "not_selected" if None in pair else "examined" if pair in examined else "not_examined"})
    n=len(predicted);rank_agreement=None
    if n>1:
        a,b=_ranks(predicted),_ranks(gold)
        if len(set(a))>1 and len(set(b))>1:rank_agreement=statistics.correlation(a,b)
    comparisons=[(predicted[i]-predicted[j],gold[i]-gold[j]) for i in range(n) for j in range(i) if gold[i]!=gold[j]]
    return {"graph_id":graph["graph_id"],"case_count":len(cases),"cases":cases,
        "relation_confusion":{"tp":tp,"fp":fp,"fn":fn,"tn":tn},
        "relation_precision":tp/(tp+fp) if tp+fp else None,"relation_recall":tp/(tp+fn) if tp+fn else None,
        "jointly_scored_cases":n,"score_mae":statistics.mean(abs(a-b) for a,b in zip(predicted,gold)) if n else None,
        "relation_type_agreement":statistics.mean(relation_matches) if relation_matches else None,
        "dimension_mae":{k:{"count":len(v),"mae":statistics.mean(v) if v else None} for k,v in dimension_errors.items()},
        "score_spearman":rank_agreement,
        "pairwise_order_agreement":statistics.mean(1 if a*b>0 else .5 if a==0 else 0 for a,b in comparisons) if comparisons else None,
        "calibration_label_overlap":False,"limitations":["Expert disagreements are reported, not silently majority-voted away.",
            "Score metrics are conditional on both judges scoring the relation; report coverage/abstention alongside them.",
            "A stage edge can aggregate multiple papers; inspect exact witness applicability in the annotation protocol.",
            "No success probabilities or field-wide completeness are established by this audit."]}


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
