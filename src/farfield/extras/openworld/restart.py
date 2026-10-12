"""CleanRestartProjection: what a pivoted project is allowed to carry forward.

The cheapest way to make a project pivot fail is to perform it and then hand
the model the same fifty-turn transcript that produced the dead direction. The
narrative anchors it: the "new" hypotheses come back rhymed with the old ones,
the abandoned mechanism reappears under a new name, and the pivot bought
nothing.

A restart therefore keeps facts and drops story:

    kept     contract, verified literature, admitted WORLD evidence,
             live contradictions, compact failure lessons, hard constraints,
             remaining budget
    dropped  speculative hypothesis chains, retired mechanism narratives,
             prompt residue from the failed lineage, per-turn commentary

Dropping is not deletion. The event log is append-only and the evidence stays
in canonical state; this is a *projection* for the next worker's context.
"""

from __future__ import annotations

from typing import Any, Mapping

from .contract import contract_from_state
from .quality import ReviewRecord, open_reviews
from .resources import budget_state


MAX_FAILURE_LESSONS = 8


def failure_lesson(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compact a failure into something reusable without re-anchoring on it.

    A lesson that only says "this failed" turns into a permanent taboo. A lesson
    that says when it applies — and when it does not — can be reconsidered.
    """
    reason = str(row.get("reason") or row.get("why") or row.get("objection") or "")
    return {
        "what_failed": str(row.get("action_type") or row.get("class") or "design"),
        "why_it_failed": reason,
        "applies_when": str(row.get("applies_when") or "") or (
            f"a design depends on: {reason}" if reason else "unknown"),
        "does_not_apply_when": str(row.get("does_not_apply_when") or "")
        or "the dependency named above is removed or independently established",
        "theory_id": str(row.get("theory_id") or ""),
        "epistemic": "GENERATED",
        "status": "lesson_not_evidence",
    }


def clean_restart_projection(state: Any, *, reason: str = "") -> dict[str, Any]:
    """Build the post-pivot context. Facts survive; speculation does not."""
    contract = contract_from_state(state)
    world_evidence = [
        {
            key: row.get(key)
            for key in (
                "evidence_id", "world_id", "world_digest", "experiment_digest",
                "outcome", "verdict", "theory_id", "question_id", "claim",
                "treatment", "control", "sample_size", "replication_of",
                "reproduction_ok", "identity_ok", "transfer_ok",
                "study_stage", "promotion_ok", "cannot_corroborate", "failure_diagnosis",
                "sample_size_basis", "inference_link", "validity",
            )
            if row.get(key) is not None
        }
        for row in getattr(state, "evidence_records", []) or []
        if str(row.get("epistemic") or "") == "WORLD"
        and row.get("attested")
        and not row.get("invalidated") and not row.get("promotion_refused")
    ]
    literature = [
        {
            "id": row.get("id"),
            "question_id": row.get("question_id"),
            "citations": list(row.get("citations") or []),
        }
        for row in getattr(state, "artifacts", []) or []
        if row.get("kind") == "literature_survey" and row.get("works")
    ]
    lessons: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in list(getattr(state, "failed_designs", []) or [])[::-1]:
        lesson = failure_lesson(row)
        key = f"{lesson['what_failed']}|{lesson['why_it_failed']}"
        if key in seen:
            continue
        seen.add(key)
        lessons.append(lesson)
        if len(lessons) >= MAX_FAILURE_LESSONS:
            break
    unresolved = [
        ReviewRecord.from_dict(row).to_dict()
        for row in open_reviews(getattr(state, "reviews", []) or [])
        if ReviewRecord.from_dict(row).severity in {"fatal", "major"}
    ]
    # An objection the direction could not answer is the most useful lesson the
    # direction produced, whether or not a design was ever logged as failed.
    for row in unresolved:
        if len(lessons) >= MAX_FAILURE_LESSONS:
            break
        lesson = failure_lesson({
            "action_type": row.get("repair_type") or "design",
            "reason": row.get("objection"),
            "theory_id": row.get("theory_id"),
            "applies_when": f"a design rests on {row.get('dimension')} being adequate",
            "does_not_apply_when": (
                f"{row.get('dimension')} is established independently, or the "
                "claim no longer depends on it"),
        })
        key = f"{lesson['what_failed']}|{lesson['why_it_failed']}"
        if key not in seen:
            seen.add(key)
            lessons.append(lesson)
    return {
        "kind": "clean_restart_projection",
        "reason": str(reason or ""),
        "project_epoch": int(getattr(state, "project_epoch", 0) or 0),
        "research_contract": contract.to_dict(),
        "root_question": contract.root_question,
        "verified_literature": literature,
        "world_evidence": world_evidence,
        "contradictions": [dict(row) for row in getattr(state, "contradictions", []) or []],
        "failure_lessons": lessons,
        "unresolved_objections": unresolved,
        "hard_constraints": list(contract.hard_constraints),
        "remaining_resources": dict(budget_state(state).get("remaining") or {}),
        "budget_pressure": float(budget_state(state).get("budget_pressure") or 0),
        "dropped": [
            "speculative_hypothesis_chains",
            "retired_mechanism_narratives",
            "failed_lineage_prompt_residue",
            "per_turn_commentary",
        ],
        "note": (
            "Dropped context is not deleted: the event log keeps it. This is the "
            "projection the next worker sees, so a retired direction cannot "
            "anchor the restart."
        ),
    }
