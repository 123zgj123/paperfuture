"""Bounded knowledge acquisition for speculative ideas, never evidence promotion.

Researchers and critics name decision-relevant uncertainties. The controller
deduplicates unchanged questions and reserves follow-up budget; it does not
limit a lineage to a fixed number of different inquiries. No promotion occurs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def text(value: Any, limit: int = 1800) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def strings(value: Any, limit: int = 12) -> list[str]:
    values = [value] if isinstance(value, str) else value if isinstance(value, (list, tuple)) else []
    return list(dict.fromkeys(s for item in values if (s := text(item))))[:limit]


def get(state: Any, key: str, default: Any = None) -> Any:
    return state.get(key, default) if isinstance(state, Mapping) else getattr(state, key, default)


OBJECTION_ACTIONS = {
    "knowledge_gap": "literature", "mechanism_gap": "literature",
    "prior_overlap": "literature", "contradiction": "design",
    "design_gap": "design", "implementation_gap": "implementation",
    "resource_gap": "capability", "impact_uncertain": "design",
}
FAILURE_OBJECTIONS = {
    "mechanism_unsupported": "mechanism_gap", "missing_structure": "mechanism_gap",
    "low_novelty": "prior_overlap", "unrealistic_assumption": "knowledge_gap",
    "causal_identifiability": "design_gap", "unfalsifiable": "design_gap",
    "competing_explanation_unresolved": "design_gap",
}


def investment_packet(idea: Mapping[str, Any], critique: Mapping[str, Any],
                      grounding: Mapping[str, Any], state: Any) -> dict[str, Any]:
    """Keep conditional value and uncertainty separate; never invent a score."""
    opportunity = idea.get("opportunity") or {}
    opportunity = opportunity if isinstance(opportunity, Mapping) else {}
    evidence_ids = {str(e.get("evidence_id")) for e in get(state, "evidence_records", [])
                    if e.get("epistemic") == "WORLD" and e.get("attested") and not e.get("invalidated")}
    objections = []
    for raw in (critique.get("objections") or [])[:12] if isinstance(critique.get("objections"), list) else []:
        if not isinstance(raw, Mapping) or not text(raw.get("claim")):
            continue
        kind = text(raw.get("kind"))
        kind = kind if kind in OBJECTION_ACTIONS else "knowledge_gap"
        refs = strings(raw.get("evidence_ids"))
        objections.append({"kind": kind, "claim": text(raw.get("claim")),
                           "check": text(raw.get("check")),
                           "source_ids": strings(raw.get("source_ids")),
                           "evidence_ids": [e for e in refs if e in evidence_ids],
                           "unresolved_evidence_ids": [e for e in refs if e not in evidence_ids],
                           "reference_status": "identity_only_not_semantic_validation",
                           "epistemic": "GENERATED"})
    if not objections:
        kind = FAILURE_OBJECTIONS.get(text(critique.get("root_failure_cause")))
        if kind:
            objections.append({"kind": kind, "claim": text(critique.get("strongest_negative_signal"))
                               or text(idea.get("mechanism")), "check": "",
                               "source_ids": [], "evidence_ids": [], "epistemic": "GENERATED"})
    if not objections and grounding.get("status") == "literature_gap":
        objections.append({"kind": "knowledge_gap", "claim": text(idea.get("mechanism")),
                           "check": "", "source_ids": [], "evidence_ids": [], "epistemic": "GENERATED"})
    choice = critique.get('priority_objection')
    selected = (objections[choice] if type(choice) is int and 0 <= choice < len(objections)
                else next((r for r in objections if r.get('check')), objections[0] if objections else {}))
    first = selected
    kind = OBJECTION_ACTIONS.get(first.get("kind"), "design")
    question = text(first.get("check")) or text(first.get("claim")) or text(idea.get("falsifier"))
    return {
        "version": "bounded-inquiry-v1", "epistemic": "GENERATED",
        "decision": "investigate" if kind == "literature" else
                    "awaiting_capability" if kind == "capability" else "develop_test",
        "conditional_value": text(opportunity.get("conditional_impact")) or text(idea.get("why_it_may_matter")),
        "mechanism_basis": {"grounding_status": grounding.get("status", "unchecked"),
                            "assumptions": strings(opportunity.get("assumptions")),
                            "confidence": "uncalibrated"},
        "discriminating_check": text(opportunity.get("cheapest_discriminating_test")) or text(idea.get("falsifier")),
        "objections": objections,
        "selection_basis": "critic_proposed_priority_not_validated" if type(choice) is int and 0 <= choice < len(objections)
                           else "first_actionable_check_unranked",
        "selection_reason": text(critique.get('priority_reason')),
        "next_check": {"kind": kind, "question": question,
                       "on_support": "formulate_a_discriminating_test_not_a_confirmed_claim",
                       "on_refutation": "revise_the_named_assumption_preserving_the_record",
                       "on_inconclusive": "retain_uncertainty_and_reassess_cost"},
        "stop_rule": text(opportunity.get("stop_rule")),
        "resume_when": strings(critique.get("required_evidence")) or strings(idea.get("uncertainties")),
        "cost_status": "not_measured", "scientific_verdict": "not_assessed",
    }


def inquiry_request(idea_row: Mapping[str, Any], feedback: Mapping[str, Any]) -> dict[str, Any]:
    packet = feedback.get("investment") or {}
    check = packet.get("next_check") or {}
    requested = (idea_row.get('idea') or {}).get('knowledge_request') or {}
    if text(requested.get('question')) and text(requested.get('decision_relevance')):
        check = {**requested, 'kind': 'literature'}
    if check.get("kind") != "literature" or not text(check.get("question")):
        return {}
    row = {"idea_id": str(idea_row["id"]), "lineage_id": str(idea_row.get("lineage_id") or "L1"),
           "question": text(check["question"]), "feedback_id": str(feedback.get("id") or ""),
           "project_epoch": int(idea_row.get("project_epoch") or 0)}
    row["id"] = "inquiry-" + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
    row.update({"on_support": check.get("on_support", "reassess"), "on_refutation": check.get("on_refutation", "revise"),
                "on_inconclusive": check.get("on_inconclusive", "retain_uncertainty"),
                "estimated_model_calls": 5, "followup_call_reserve": 2,
                "epistemic": "GENERATED"})
    return row


def inquiry_progress(state: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    epoch = int(get(state, "project_epoch", 0))
    rows = [r for r in get(state, "artifacts", []) if int(r.get("project_epoch") or 0) == epoch]
    surveys = [r for r in rows if r.get("kind") == "literature_survey"
               and (r.get("knowledge_request") or {}).get("id") == request["id"]]
    attempts = [r for r in rows if r.get("kind") == "action_attempt"
                and r.get("action_type") == "SURVEY" and r.get("target") == request["id"]]
    prior_questions = [r for r in rows if r.get('kind') == 'literature_survey'
        and (r.get('knowledge_request') or {}).get('lineage_id') == request['lineage_id']
        and text((r.get('knowledge_request') or {}).get('question')).casefold() == text(request['question']).casefold()]
    costs = sorted(float(r['measured_cost']['api_calls']) for r in rows
        if r.get('kind') == 'action_attempt' and r.get('action_type') == 'SURVEY'
        and r.get('inquiry_lineage_id') == request['lineage_id']
        and isinstance((r.get('measured_cost') or {}).get('api_calls'), (int, float))
        and 0 < r['measured_cost']['api_calls'] < float('inf'))
    estimate = max(2, costs[(3 * len(costs) - 1) // 4]) if costs else request['estimated_model_calls']
    remaining = (get(state, "resources", {}) or {}).get("remaining", {}).get("api_calls")
    enough = remaining is None or remaining >= estimate + request["followup_call_reserve"]
    status = ("sources_returned" if surveys else "attempted_without_sources" if attempts else
              "question_already_investigated" if prior_questions else
              "budget_insufficient" if not enough else "ready")
    return {"request": dict(request), "status": status,
            "estimated_model_calls": estimate,
            "cost_basis": "recent_lineage_survey_costs" if costs else "initial_call_estimate",
            "prior_survey_ids": [r['id'] for r in prior_questions],
            "survey_ids": [r["id"] for r in surveys],
            "assessment": "sources_require_interpretation" if surveys else "under_assessed_not_refuted"}


def knowledge_move(state: Any, row: Mapping[str, Any], feedback: Mapping[str, Any]) -> dict[str, Any] | None:
    request = inquiry_request(row, feedback)
    if not request or inquiry_progress(state, request)["status"] != "ready":
        return None
    return {"action": "SURVEY", "target": request["id"], "reason": "resolve_named_research_uncertainty",
            "extra": {"knowledge_request": request, "discovery_idea_id": row["id"],
                      "lineage_id": request["lineage_id"], "extract_trajectories": True,
                      "extraction_budget": {"papers": 2, "annotated_edges": 0},
                      "claim": text((row.get("idea") or {}).get("central_claim"))}}


def inquiry_context(state: Any, parent: Mapping[str, Any] | None,
                    feedback: Mapping[str, Any] | None) -> dict[str, Any]:
    request = inquiry_request(parent, feedback or {}) if parent else {}
    return inquiry_progress(state, request) if request else {}


def inquiry_is_due(state: Any, request: Mapping[str, Any]) -> bool:
    """Recompute authority from the current idea and feedback, not action flags."""
    from .discovery import discovery_records

    ideas, feedback = discovery_records(state)
    row = next((r for r in ideas if r.get("id") == request.get("idea_id")), None)
    if not row:
        return False
    expected = inquiry_request(row, feedback.get(row["id"], {}))
    return bool(expected and dict(request) == expected
                and inquiry_progress(state, expected)["status"] == "ready")
