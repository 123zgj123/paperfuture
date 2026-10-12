"""Conditional experiment selection over declared numerical predictions.

Interval compatibility is a planning aid, not posterior probability, power,
or scientific confirmation. Predictions may be wrong or generated after data
were seen. Unknown mechanisms must remain possible outside the candidate set.
"""
from __future__ import annotations

import copy
import math
from itertools import combinations
from typing import Any, Mapping, Sequence

from ...services.experiments import fingerprint


def _number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def design_feedback(candidates: Sequence[Mapping[str, Any]],
                    observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank unmeasured designs by disjoint prediction pairs per declared cost.

    Each candidate names a frozen world, a complete execution object and a
    prediction interval for treatment-minus-control under every hypothesis.
    Matching, attested, reproduced executions check forecasts, never eliminate
    hypotheses. Uncalibrated interval misses do not license that inference.
    All observed matches are excluded from next-design suggestions, even if
    their measurements are not yet verified (suggest verification instead).
    """
    designs = copy.deepcopy(list(candidates))
    if not designs:
        return {"status": "missing_predictions", "ranked_designs": [], "epistemic": "GENERATED"}
    if len(designs) > 32:
        raise ValueError("design_planner_limit_32_candidates")
    names: set[str] | None = None
    ids: set[str] = set()
    for row in designs:
        if not isinstance(row, Mapping):
            raise ValueError("design_must_be_an_object")
        key, predictions = row.get("id"), row.get("predictions")
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError("design_id_missing_or_duplicate")
        ids.add(key)
        if any(not isinstance(row.get(key), str) or not row[key] for key in ("world_id", "world_digest")):
            raise ValueError("design_requires_frozen_world_identity")
        if not isinstance(row.get("experiment"), Mapping) or not row["experiment"].get("metric"):
            raise ValueError("design_requires_execution_and_metric")
        if not _number(row.get("cost")) or row["cost"] <= 0:
            raise ValueError("design_cost_must_be_positive_finite")
        if not isinstance(predictions, Mapping) or len(predictions) < 2:
            raise ValueError("design_requires_competing_predictions")
        if len(predictions) > 128:
            raise ValueError("design_planner_limit_128_hypotheses")
        if any(not isinstance(name, str) or not name for name in predictions):
            raise ValueError("hypothesis_names_must_be_nonempty_strings")
        for interval in predictions.values():
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or not all(_number(v) for v in interval) or interval[0] > interval[1]):
                raise ValueError("predictions_require_finite_ordered_intervals")
        if names is not None and names != set(predictions):
            raise ValueError("all_designs_must_predict_the_same_hypotheses")
        names = set(predictions)
    remaining = set(names or ())
    compatible = set(remaining)
    measured, used, ignored, exclusions = set(), [], [], []
    for evidence in observations:
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("registration"), Mapping):
            continue
        matching = [row for row in designs if
            evidence.get("world_id") == row["world_id"]
            and evidence.get("world_digest") == row["world_digest"]
            and (evidence.get("registration") or {}).get("execution") == row["experiment"]
            and not evidence.get("study_setting_id")]
        if not matching:
            continue
        eid = str(evidence.get("evidence_id") or "")
        valid = (eid and evidence.get("epistemic") == "WORLD" and evidence.get("attested")
                 and not evidence.get("invalidated") and evidence.get("execution_status") == "ran"
                 and not evidence.get("promotion_refused")
                 and evidence.get("replay_ok") is True
                 and _number(evidence.get("treatment")) and _number(evidence.get("control")))
        if (not evidence.get("invalidated") and evidence.get("epistemic") == "WORLD"
                and evidence.get("attested") and evidence.get("execution_status") == "ran"):
            measured.update(row["id"] for row in matching)
        if not valid:
            ignored.append(eid)
            continue
        delta = evidence["treatment"] - evidence["control"]
        if not math.isfinite(delta):
            ignored.append(eid)
            continue
        used.append(eid)
        for row in matching:
            incompatible = {name for name, (low, high) in row["predictions"].items()
                            if not low <= delta <= high}
            compatible -= incompatible
            exclusions.append({"design_id": row["id"], "evidence_id": eid,
                               "observed_difference": delta,
                               "incompatible_predictions": sorted(incompatible)})
    ranking = []
    for row in designs:
        if row["id"] in measured:
            continue
        disjoint = sum(row["predictions"][a][1] < row["predictions"][b][0]
                       or row["predictions"][b][1] < row["predictions"][a][0]
                       for a, b in combinations(sorted(remaining), 2))
        score = disjoint / row["cost"]
        if not math.isfinite(score):
            raise ValueError("design_cost_yields_nonfinite_score")
        ranking.append({"id": row["id"], "separated_pairs": disjoint,
                        "score": score, "declared_cost": row["cost"],
                        "design_digest": fingerprint(row)})
    ranking.sort(key=lambda row: (-row["score"], row["id"]))
    violations = any(row['incompatible_predictions'] for row in exclusions)
    status = ("forecast_revision_required" if violations else
              "discriminating_design_available" if ranking and ranking[0]["score"] > 0 else
              "no_discriminating_design")
    next_step = {
        "forecast_revision_required": "check_forecast_implementation_measurement_and_scope_before_revising_hypotheses",
        "no_discriminating_design": "change_observable_or_expand_hypotheses",
        "discriminating_design_available": "consider_ranked_designs",
    }[status]
    return {"version": "interval-design-v2", "epistemic": "GENERATED", "status": status,
            "active_hypotheses": sorted(remaining),
            "interval_compatible_hypotheses": sorted(compatible),
            "hypothesis_refutations": [], "forecast_calibration": "unverified",
            "ranking_requires_forecast_review": violations, "ranked_designs": ranking,
            "used_evidence_ids": sorted(set(used)), "unusable_evidence_ids": sorted(set(ignored)),
            "prediction_checks": exclusions,
            "next_step": "verify_observations" if remaining and ignored else next_step,
            "candidate_set_completeness": "unverified",
            "scientific_completion": False,
            "scope": "conditional_on_declared_intervals_and_candidate_set; not_a_scientific_verdict",
            "score_basis": "nonoverlapping_prediction_pairs_per_declared_cost; not_information_gain"}


def research_design_feedback(state: Any) -> list[dict[str, Any]]:
    """Expose the same planner in the journal read by idea and experiment workers."""
    from .discovery import discovery_view, idea_of

    result = []
    for row in discovery_view(state).get("ideas", []):
        candidates = idea_of(row).design_candidates
        if not candidates:
            continue
        try:
            feedback = design_feedback(candidates, getattr(state, "evidence_records", []))
        except (TypeError, ValueError, KeyError) as exc:
            feedback = {"status": "invalid_design_predictions", "reason": str(exc), "epistemic": "GENERATED"}
        result.append({"idea_id": row.get("id"), **feedback})
    return result
