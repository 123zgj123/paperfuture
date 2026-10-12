"""Discovery workers: the model-backed half of the Discovery Runtime.

IDEATE writes one complete idea — from a remote bootstrap on the first hop, and
afterwards from the previous idea and its feedback packet. CRITIQUE validates
it: deterministic structural checks (V1), a literature-grounding check, and
separate model calls for a research-world rollout (V2) and a critic. The optional
fast mode combines those two calls. Neither is an executed experiment. The
feedback packet is composed deterministically from those parts
(`discovery.compose_feedback`).

Both return a `CandidateResult` whose only proposal is one `ArtifactAdded`. The
controller's admission boundary accepts nothing else from these actions and
stamps what it accepts GENERATED. A worker here sees a read-only view of the
state and cannot write it.

Artifacts are written as files too, because an idea, its research plan and its
experiment design are the product of a discovery run, not an intermediate step
before a probe. Files are never deleted: a later failed validation adds a
feedback packet, it does not remove the idea it judged.
"""

from __future__ import annotations

import json
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .discovery import (
    BRANCH_DIVERSIFICATION,
    CRITIQUE,
    EVOLVE,
    FAILURE_MODES,
    FEEDBACK_KIND,
    IDEA_KIND,
    IDEATE,
    MODES,
    NO_FAILURE,
    OPERATORS,
    PROMISING,
    REMOTE_BOOTSTRAP,
    REMOTE_BRANCH,
    REMOTE_ESCAPE,
    CompleteIdea,
    compose_feedback,
    current_policy,
    discovery_summary,
    discovery_view,
    evidence_route,
    idea_of,
    lineages,
    literature_grounding,
    normalize_rollout,
    root_question,
    structural_validation,
)
from .events import make_event
from .kernel import CandidateResult, readonly_view
from .remote import as_remote_result

DISCOVERY_DIR = "discovery"

SYSTEM_IDEATE = (
    "You propose one complete research idea. Return exactly one JSON object with the keys: "
    "title, research_problem, central_claim, mechanism, motivation, why_it_may_matter, "
    "novelty_hypothesis, competing_explanation, falsifier, validation_direction, "
    "optional_experiment_direction, grounding (list of strings; cite only ids given under "
    "verified_literature, never invent a citation), uncertainties (list), claim_type (one of "
    "conceptual, theoretical, empirical_observational, mechanistic_intervention), "
    "structural_correspondence, preserved_from_parent (list), changed_from_parent (list), "
    "addresses. Everything you write is a GENERATED proposal: never state that anything has "
    "been shown, measured or verified. The falsifier names an outcome that would refute the "
    "claim. The competing explanation is the strongest rival account of the same phenomenon. "
    "Do not argue novelty from conceptual distance. Two concepts side by side are not an idea."
)
from .experiment import PROTOCOL_HELP
SYSTEM_IDEATE += " " + PROTOCOL_HELP
from .response import RESPONSE_HELP
SYSTEM_IDEATE += RESPONSE_HELP
from .exploratory import EXPLORATION_HELP
SYSTEM_IDEATE += EXPLORATION_HELP
SYSTEM_IDEATE += (
    " When considering transfer, return transfer_analysis: {source_mechanism, "
    "source_ids:[], target_mapping, required_assumptions:[], failure_conditions:[], "
    "discriminating_test, landing_questions:[], implementation_correspondence, "
    "performance_test, mechanism_test, rejected_frames:[]}. Mark missing sources explicitly. "
    "Explain a shared relation and a possible breaking difference, not just an analogy. "
    "After a knowledge inquiry, compare new passages against the previous objection; "
    "revise the mechanism or retain uncertainty. Do not equate citation identity with "
    "support. Preserve conflicting findings and failed historical routes. "
    "To resolve a specific knowledge gap before implementation, return knowledge_request: "
    "{question, decision_relevance, on_support, on_refutation, on_inconclusive}. This can "
    "request source-mechanism evidence, nearby target methods or failed transfer attempts. "
    "Do not request a generic topic summary. The runtime will search and return sources; "
    "interpret them before deciding whether to implement or revise. This request is optional "
    "and does not require a critic's permission. Omit exploration when choosing literature "
    "as the next action; do not precommit code that ignores the incoming evidence. "
    "Missing history does not force rejection; make the cheapest informative check. "
)
SYSTEM_IDEATE += (
    " Also return opportunity: {source_ids:[], technical_trajectory, unresolved_bottleneck, "
    "proposed_departure, why_underexplored, conditional_impact, assumptions:[], "
    "cheapest_discriminating_test, stop_rule}. Explain who benefits and by which causal path "
    "IF the hypothesis holds. Rare model output, lexical distance, and unfamiliarity are "
    "not low scientific prior probability or impact. Do not invent calibrated probabilities. "
    "Treat missing evidence as uncertainty to investigate, not automatic rejection or success. "
    "Check dimensional consistency, limiting cases, controls and competing explanations "
    "before expensive work; name which check could overturn the idea. "
    "A fallback corpus is a brainstorming source, not authority. You may reject all its frames. "
    "Follow dated technical changes when available; otherwise state that the historical "
    "trajectory is missing and identify the literature needed to establish it. ")
SYSTEM_IDEATE += (
    " The mission and its hard constraints are immutable; research questions may change. "
    "When recorded observations or a live blocker invalidate a question's assumption, you "
    "may additionally propose question_transition: {id, text, parent_question_id, "
    "triggering_evidence_ids: [], triggering_blocker_ids: [], invalidated_assumption, "
    "rationale, mission_link, remaining_mission_gaps: []}. Cite only IDs in current state. "
    "Give the new question a fresh ID. State how Q2 serves the same mission and what it "
    "leaves unanswered. Do not merely restate the mission. This is a GENERATED proposal, "
    "not proof of mission alignment or completion. Preserve Q1 and its original experiment "
    "verdict; Q2 requires a new registered experiment and fresh observations. If no recorded "
    "trigger exists, omit question_transition.")
SYSTEM_IDEATE += (
    " To propose a discriminating experiment, also return predictions as an object mapping "
    "hypothesis names to {observable: the SAME measured variable, direction: "
    "increase|decrease|unchanged}. Different wording is not a different prediction. "
    "If predictions are unknown leave predictions empty; never invent discrimination. "
    "These are conditional forecasts, not evidence or a statistical power assessment.")
SYSTEM_IDEATE += (
    " Optionally provide design_candidates: a list of {id, world_id, world_digest, "
    "cost: a positive finite NUMBER in shared cost units (never a label such as cheap), "
    "experiment: complete executable experiment object, predictions: {hypothesis_name: "
    "[lower, upper]}}. Intervals predict treatment MINUS control for that experiment and "
    "frozen world, including observation variability. All designs must cover the same rival "
    "hypotheses. Use only known world identities. Do not fabricate numerical predictions; "
    "omit this optional field when bounds are unknown. These generated intervals are NOT "
    "confidence intervals. current_research_state.experiment_design_feedback checks them "
    "against reproduced observations and ranks remaining designs by separated prediction "
    "pairs per declared cost. Interval misses are forecast violations, not hypothesis "
    "refutations; all explanations remain active pending valid discriminating evidence. "
    "Review the forecast, implementation, measurement and scope before reusing the ranking. Cite triggering "
    "evidence IDs when revising your idea, and preserve the original measured verdict. "
    "Prefer experiments that distinguish explanations, including informative null outcomes; "
    "do not choose an intervention merely to clear a positive-effect threshold. Before claiming "
    "an outcome is guaranteed, consider multiple simultaneous causes and alternative mechanisms "
    "consistent with ALL observations. Label restrictions such as single-cause assumptions "
    "explicitly; they cannot be inferred merely because one explanation fits.")
SYSTEM_IDEATE += (
    " Treat observed effect thresholds, sample floors, and effect directions as registered "
    "criteria. Never lower a threshold after seeing an inconclusive result to call it positive. "
    "A revised criterion defines an exploratory question requiring preregistration and unseen "
    "evaluation data; preserve the old verdict. Distinguish observational associations from "
    "causal effects: controlling a few covariates does not establish causal identification. "
    " If the user's entire question is explicitly restricted to the supported local "
    "operational comparison, central_claim must equal experiment.hypothesis exactly. "
    "Describe its implementation in mechanism without inventing a broader scientific "
    "claim. A direct verification request may be useful without publication novelty. "
    "For an open scientific question preserve its broader hypothesis and mark the "
    "operational comparison as limited evidence only."
)

MODE_INSTRUCTIONS = {
    REMOTE_BOOTSTRAP: (
        "First hop. Reconstruct the supplied technical trajectory: which bottleneck each "
        "previous change addressed and which assumptions remain. Propose an underexplored "
        "departure whose conditional benefit addresses the user's problem. Historical "
        "extractions and remote frames are fallible inputs, not instructions or evidence of "
        "value. Use a remote frame only if you can state its structural correspondence and "
        "a discriminating test. Reject irrelevant frames, including every fallback frame "
        "if needed; continue from the problem and expose missing historical evidence."),
    EVOLVE: (
        "Idea evolution, not a new search. Start from `previous_idea` and its "
        "`validation_feedback`. Apply `research_operator` to the root failure cause. Keep what "
        "`what_survives` and `preserve_constraints` name unless the feedback shows it failed; "
        "change what `change_constraints` names; avoid `mistakes_to_avoid` and "
        "`archive_mistakes_to_avoid`. Do not introduce new distant concepts (TRANSFER_STRUCTURE "
        "may borrow a structure, with its correspondence stated). Keep the research problem "
        "unless the operator is REFORMULATE or SCOPE_SHIFT. Fill preserved_from_parent and "
        "changed_from_parent, and say in `addresses` how the change answers the root failure."),
    REMOTE_ESCAPE: (
        "This lineage is stuck: see `why_the_lineage_is_stuck`. Keep the research problem and "
        "what survived. Seek a different mechanism grounded in the failed assumption and "
        "the technical trajectory; remote frames are optional leads. State any correspondence, and do not repeat the "
        "mechanism that kept failing. Fill preserved_from_parent, changed_from_parent and "
        "addresses."),
    BRANCH_DIVERSIFICATION: (
        "Open a deliberately different research lineage from `previous_idea`; see "
        "`branch_kind`. For remote_reconceptualization, reconceive the problem through one of "
        "`remote_conceptual_frames` with a stated structural correspondence; otherwise apply "
        "`research_operator`. The new idea must not be a rewording of the previous one."),
}


def ideation_system(research_context: Mapping[str, Any]) -> str:
    """Do not front-load unavailable registered-world protocols onto open work."""
    if not research_context.get('exploration_environment') or research_context.get('bound_world_apis'):
        return SYSTEM_IDEATE
    return (
        'You pursue the user research goal through real evidence and open programs. '
        'Return one JSON object. For initial inspection or method development, return '
        'exploration and the required result_response; scientific claims may remain provisional. '
        'Keep user intent and hard constraints; revise subquestions when observations warrant it. '
        'Use metric_feedback to locate errors: a combined score alone does not identify their cause. '
        'Separate task delivery, method performance and mechanism evidence. '
        'Never invent observations, citations, calibrated predictions or successful interventions. '
        'You may instead propose knowledge_request:{question,decision_relevance,on_support,'
        'on_refutation,on_inconclusive} to investigate a specific evidence gap. '
        'For a developed idea include title,research_problem,central_claim,mechanism,'
        'competing_explanation,falsifier,uncertainties. For mechanism transfer explain '
        'source evidence, shared structure, required assumptions, breaking differences, '
        'landing questions and separate performance and mechanism tests. No novelty or '
        'impact narrative is required for preliminary work. A knowledge request is not '
        'evidence; actual tool availability is specified by the host policy. '
        + RESPONSE_HELP + EXPLORATION_HELP)

SYSTEM_ROLLOUT = (
    "You are a GENERATED research-world simulator, not a critic and not a source of evidence. "
    "Predict how research on this idea would develop if a competent group pursued it. "
    "Return exactly one JSON object with keys: likely_next_questions, "
    "likely_competing_explanations, likely_missing_knowledge, predicted_contradictions, "
    "required_evidence, possible_research_path, dependency_structure (each a list of strings), "
    "likely_failure_point, what_would_kill_the_idea, what_result_would_strengthen_it, "
    'outcome: "DEVELOPS"|"FRAGILE"|"COLLAPSES", reason. Do not judge whether the idea is '
    "promising. Do not assign a verdict."
)

SYSTEM_CRITIQUE = (
    "You are a GENERATED critic of a research idea. You are not the research-world simulator. "
    "Judge an early idea as an early idea: a missing experiment, replication or baseline is "
    "not a failure; list what would be needed under required_evidence. Return exactly one "
    "JSON object {strongest_positive_signal, strongest_negative_signal, what_survives, "
    "what_failed, root_failure_cause, strongest_competing_explanation, repairable (bool), "
    "blocking (bool: true only if the idea is not worth pursuing as research, "
    "never because a fixture is missing or a protocol is not yet runnable), "
    "suggested_operator, preserve_constraints, change_constraints, mistakes_to_avoid, "
    "new_questions, required_evidence (lists of strings where plural), novelty_assessment, "
    "importance_assessment}. Do not treat low novelty as an execution ban. "
    "root_failure_cause is one of "
    + ", ".join(list(FAILURE_MODES) + [NO_FAILURE])
    + ". suggested_operator is one of " + ", ".join(OPERATORS) + "."
)

SYSTEM_CRITIQUE_FAST = (
    "You play two GENERATED roles, and neither certifies truth. (1) Research-world simulator: "
    "predict how research on this idea would develop if a competent group pursued it. "
    "(2) Critic: diagnose the idea as a research hypothesis. Judge an early idea as an early "
    "idea: a missing experiment, replication or baseline is not a failure; list what would be "
    "needed under required_evidence. Return exactly one JSON object "
    '{"rollout": {likely_next_questions, likely_competing_explanations, likely_missing_knowledge, '
    "predicted_contradictions, required_evidence, possible_research_path, dependency_structure "
    "(each a list of strings), likely_failure_point, what_would_kill_the_idea, "
    'what_result_would_strengthen_it, outcome: "DEVELOPS"|"FRAGILE"|"COLLAPSES", reason}, '
    '"critique": {strongest_positive_signal, strongest_negative_signal, what_survives, '
    "what_failed, root_failure_cause, strongest_competing_explanation, repairable (bool), "
    "blocking (bool: true only if the idea is not worth pursuing as research, "
    "never because a fixture is missing or a protocol is not yet runnable), "
    "suggested_operator, preserve_constraints, change_constraints, mistakes_to_avoid, "
    "new_questions, required_evidence (lists of strings where plural), novelty_assessment, "
    "importance_assessment}}. Do not treat low novelty as an execution ban. "
    "root_failure_cause is one of "
    + ", ".join(list(FAILURE_MODES) + [NO_FAILURE])
    + ". suggested_operator is one of " + ", ".join(OPERATORS) + "."
)


MECHANISM_CHECKS = (
    " Check omitted combinations of mechanisms and assumptions behind claimed guaranteed outcomes. "
    "A useful experiment can return a null or adverse result; assess its ability to distinguish "
    "explanations rather than its ability to cross a positive-effect threshold. "
    "Also return objections:[{kind: knowledge_gap|mechanism_gap|prior_overlap|contradiction|"
    "design_gap|implementation_gap|resource_gap|impact_uncertain, claim, check, source_ids:[], "
    "evidence_ids:[]}]. Each check names the smallest question or observation that would "
    "change the decision. Missing knowledge triggers investigation; missing tools triggers "
    "a capability request. Optionally return priority_objection as a zero-based index and "
    "priority_reason explaining which check should precede the others; this is a fallible "
    "decision proposal, not an objective value score. Unfamiliarity and uncertain impact "
    "are not falsification. "
    "Keep conditional value, mechanism support, test discrimination and execution cost "
    "separate. Do not assert a calibrated probability or recommend stopping on rhetoric alone.")
SYSTEM_CRITIQUE += MECHANISM_CHECKS
SYSTEM_CRITIQUE_FAST += MECHANISM_CHECKS


class DiscoveryWorkers:
    """IDEATE, CRITIQUE and the judging half of REVIEW over a model client.

    Nothing is called at construction. ``panel_clients`` binds individual panel
    roles (examiner, advocate, skeptic, adjudicator) to other models; a role left
    out is served by ``client``, and the record says so.
    """

    def __init__(self, client: Any = None, workspace: Path | None = None, *,
                 remote_search: Callable[..., Any] | None = None, corpus_id: str = "",
                 panel_clients: Mapping[str, Any] | None = None) -> None:
        self.client = client
        self.panel_clients = dict(panel_clients or {})
        self.workspace = Path(workspace) if workspace is not None else None
        self.corpus_id = str(corpus_id or "")
        self._remote = remote_search

    def remote_search(self, **kwargs: Any) -> Any:
        if self._remote is None:
            from .remote import ConceptSpaceSearch

            self._remote = ConceptSpaceSearch(self.corpus_id)
        return self._remote(**kwargs)

    def executors(self) -> dict[str, Callable[..., CandidateResult]]:
        return {IDEATE: self.ideate, CRITIQUE: self.critique, "REVIEW": self.review}

    def _workspace(self, kwargs: Mapping[str, Any]) -> Path | None:
        value = kwargs.get("workspace") or self.workspace
        return Path(value) if value is not None else None

    def ideate(self, action: Any, state: Any, **kwargs: Any) -> CandidateResult:
        return ideate(action, state, client=self.client, workspace=self._workspace(kwargs),
                      remote_search=self.remote_search, projection=kwargs.get("projection"))

    def critique(self, action: Any, state: Any, **kwargs: Any) -> CandidateResult:
        return critique(action, state, client=self.client, workspace=self._workspace(kwargs))

    def review(self, action: Any, state: Any, **kwargs: Any) -> CandidateResult:
        from .judgment import ROLES, review

        clients = {role: self.panel_clients.get(role) or self.client for role in ROLES}
        return review(action, state, clients=clients, workspace=self._workspace(kwargs))


# ---------------------------------------------------------------------------
# IDEATE
# ---------------------------------------------------------------------------


def _refused(reason: str, *, status: str = "blocked", **extra: Any) -> CandidateResult:
    """A discovery attempt that produced nothing. It proposes no event: nothing
    was learned about any question, and nothing may be blocked on its account."""
    return CandidateResult(status=status, events=[], extras={"reason": reason, **extra})


def _service_unavailable(error: Exception) -> CandidateResult:
    from ..llm import quota_exhausted
    from .resources import local_model_budget_meter

    text = str(error)
    record = getattr(error, "record", None)
    meter = local_model_budget_meter(error)
    if meter:
        return CandidateResult(status='budget_exhausted', events=[], extras={
            'reason': text, 'budget_meter': meter, 'budget_basis': 'typed_client_envelope'})
    return CandidateResult(status="service_unavailable", events=[], extras={
        "reason": text, "quota_exhausted": quota_exhausted(text),
        "service": str(getattr(record, "missing_capability", "") or "llm_endpoint")})


def parse_json_object(text: str) -> dict[str, Any]:
    body = str(text or "").strip()
    if body.startswith("```"):
        body = "\n".join(line for line in body.splitlines() if not line.startswith("```"))
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the completion")
    payload = json.loads(body[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("the completion is not a JSON object")
    return payload


def _literature(state: Any) -> dict[str, Any]:
    from ..livefeed import FreshWork

    works: dict[str, dict[str, str]] = {}
    transitions: list[Mapping[str, Any]] = []
    source_readings, passages, trajectory_sources, technical_maps = [], [], [], []
    for row in (state.get("artifacts") or []):
        if row.get("kind") != "literature_survey":
            continue
        if int(row.get('project_epoch') or 0) != int(state.get('project_epoch') or 0):
            continue
        transitions.extend(row.get("research_transitions") or [])
        trajectory_sources.extend(row.get('trajectory_sources') or [])
        if row.get('source_check'):
            source_readings.append(row['source_check'])
        if row.get('technical_map') or row.get('technical_map_draft'):
            technical_maps.append(row.get('technical_map') or row['technical_map_draft'])
        for raw in row.get("works") or []:
            try:
                work = FreshWork.from_dict(raw)
            except (TypeError, ValueError, KeyError):
                continue
            if work.cite_id():
                checks = {r.get('cite_id'): r.get('status') for r in row.get('verification') or []}
                relevance = row.get('retrieval_relevance') or {}
                if checks and checks.get(work.cite_id()) != 'verified':
                    continue
                if 'usable' in relevance and work.cite_id() not in relevance['usable']:
                    continue
                works.setdefault(work.cite_id(), {"cite_id": work.cite_id(), "title": work.title,
                                                  "abstract": str(work.abstract or "")[:2400],
                                                  "published": str(work.published or '')})
        passages.extend({"cite_id": s.get('cite_id'), "text": str(s.get('text') or '')[:2400],
                         "field": s.get('field'), "url": s.get('url')}
                        for s in row.get('spans') or [] if s.get('cite_id') in works)
    operators = sorted({str(t.get("operator")) for t in transitions
                        if t.get("operator") and t.get("operator") != "unknown"})
    return {"works": list(works.values()), "transitions": len(transitions),
            "operators": operators[:8],
            "source_readings": source_readings[-4:],
            "source_passages": passages[-8:],
            "trajectory_sources": trajectory_sources[-8:],
            "technical_maps": technical_maps[-2:],
            "technical_trajectory": [{k: t.get(k) for k in (
                'source_cite_id', 'target_cite_id', 'source_published', 'target_published',
                'operator', 'trigger', 'changed_component', 'preserved_component',
                'mechanism_delta', 'known_failure_conditions', 'provenance', 'epistemic')}
                for t in transitions if t.get('status') == 'extracted'][:8],
            "trajectory_basis": "citation_backed_generated_extraction_not_proof_of_progress"}


def _feedback_brief(fb: Mapping[str, Any]) -> dict[str, Any]:
    rollout = dict(fb.get("rollout") or {})
    return {key: fb.get(key) for key in (
        "verdict", "what_survives", "what_failed", "root_failure_cause",
        "strongest_competing_explanation", "preserve_constraints", "change_constraints",
        "mistakes_to_avoid", "new_questions", "required_evidence")} | {
        "investment": dict(fb.get('investment') or {}),
        "likely_failure_point": rollout.get("likely_failure_point"),
        "what_would_kill_the_idea": rollout.get("what_would_kill_the_idea")}


def _lineage_history(view: Mapping[str, Any], lineage_id: str) -> list[dict[str, Any]]:
    chain = lineages(view["ideas"]).get(lineage_id) or []
    history = []
    for row in chain:
        fb = view["feedback"].get(str(row["id"])) or {}
        history.append({"idea_id": row["id"], "hop": row.get("hop"), "mode": row.get("mode"),
                        "operator": row.get("operator") or "", "claim": idea_of(row).central_claim[:240],
                        "verdict": fb.get("verdict", "unvalidated"),
                        "root_failure_cause": fb.get("root_failure_cause", "")})
    return history


def _archive_mistakes(view: Mapping[str, Any], limit: int = 8, lineage_id: str = '') -> list[str]:
    out: list[str] = []
    allowed = {r['id'] for r in view['ideas'] if r.get('lineage_id') == lineage_id}
    for idea_id, fb in view["feedback"].items():
        if lineage_id and idea_id not in allowed:
            continue
        for item in fb.get("mistakes_to_avoid") or []:
            if item not in out:
                out.append(item)
    return out[-limit:]


def ideation_prompt(*, mode: str, root: str, operator: str = "", branch_kind: str = "",
                    parent: Mapping[str, Any] | None = None,
                    parent_fb: Mapping[str, Any] | None = None,
                    history: list[dict[str, Any]] | None = None,
                    mistakes: list[str] | None = None, remote: Mapping[str, Any] | None = None,
                    literature: Mapping[str, Any] | None = None,
                    stagnation: list[str] | None = None,
                    must_change: str = "", research_context: Mapping[str, Any] | None = None,
                    decision_request: str = "",
                    panel_verdict: Mapping[str, Any] | None = None) -> str:
    literature = dict(literature or {"works": [], "operators": []})
    payload: dict[str, Any] = {"research_intent": root, "mode": mode}
    if research_context:
        context = copy.deepcopy(dict(research_context))
        # Only replace exact duplicates. All results, negative observations,
        # retractions and response digests remain intact in result_packet.
        results = (context.get('result_packet') or {}).get('results', [])
        for entry in (context.get('local_learning') or {}).get('memory', []):
            observations = []
            for observation in entry.get('exploratory_observations', []):
                match = next((r for r in results if r.get('observation') == observation), None)
                observations.append({'result_ref': match['result_id'], 'digest': match['digest']}
                                    if match else observation)
            entry['exploratory_observations'] = observations
        if context.get('mission_contract') == context.get('research_contract') and context.get('mission_contract'):
            context.pop('mission_contract')
            context['mission_contract_ref'] = 'current_research_state.research_contract'
        payload["current_research_state"] = context
        payload["question_transition_instruction"] = (
            "A reformulation must name parent_question_id, triggering_evidence_ids and/or "
            "triggering_blocker_ids, invalidated_assumption, rationale, mission_link and "
            "remaining_mission_gaps. Structural admission does not verify semantic alignment. "
            "Use the listed questions, live observations and blockers; do not invent references.")
        payload["evidence_update_instruction"] = (
            "Use actual observations and failure diagnoses below, not only the earlier critic. "
            "In addresses name the evidence IDs, which prediction or assumption changed, and "
            "why the next experiment distinguishes the remaining explanations. A reproducible "
            "negative observation is not proof its mechanism is wrong; a pilot cannot establish "
            "generalization. Preserve unknowns, sample limits and the user's scope. If no valid "
            "observation exists, state that the revision is speculative. A new mechanism or "
            "measurement must change the research design, not merely its wording.")
    if remote:
        payload["remote_search_status"] = str(remote.get("status") or "not_requested")
        if remote.get("gap"):
            payload["remote_search_gap"] = remote["gap"]
        if remote.get("corpus_fallback"):
            payload["remote_corpus_fallback"] = {
                "requested": remote.get("requested_corpus"), "resolved": remote.get("corpus")}
    if remote and remote.get("used"):
        payload["local_research_basin"] = [row.get("label") for row in remote.get("local_basin") or []
                                           if isinstance(row, Mapping)]
        payload["remote_conceptual_frames"] = [row.get("label") for row in remote.get("frames") or []]
        payload['remote_mechanism_candidates'] = list(remote.get('frames') or [])
    elif remote and remote.get("gap"):
        payload["remote_search_instruction"] = (
            "No usable remote frame was retrieved. Continue from the research question and "
            "observed evidence; do not invent a retrieved frame or claim structural transfer.")
    payload["verified_literature"] = (
        [f"{w['cite_id']}: {w['title']}" for w in literature["works"][:8]]
        or "none retrieved: say so in grounding and do not invent citations")
    if literature.get("operators"):
        payload["historical_research_moves"] = literature["operators"]
    payload['technical_trajectory'] = literature.get('technical_trajectory') or []
    payload['trajectory_basis'] = literature.get('trajectory_basis', 'not_supplied')
    payload['source_passages'] = literature.get('source_passages') or []
    payload['source_readings'] = literature.get('source_readings') or []
    payload['trajectory_sources'] = literature.get('trajectory_sources') or []
    payload['technical_maps'] = literature.get('technical_maps') or []
    payload['dated_sources'] = [{k: w.get(k) for k in ('cite_id', 'title', 'published')}
                                for w in literature['works'][:8]]
    if parent is not None:
        payload["previous_idea"] = dict(parent.get("idea") or {})
        # Keep the full executed proposal once, in the authoritative result
        # packet. Do not truncate a negative result or summarize its contents.
        proposal = payload['previous_idea'].get('exploration')
        if proposal:
            for result in ((research_context or {}).get('result_packet') or {}).get('results', []):
                if (result.get('kind') == 'unvalidated_exploration'
                        and (result.get('observation') or {}).get('proposal') == proposal):
                    payload['previous_idea'].pop('exploration')
                    payload['previous_idea']['exploration_ref'] = result['result_id']
                    break
    if parent_fb:
        payload["validation_feedback"] = _feedback_brief(parent_fb)
    if panel_verdict:
        payload["judgment_panel_verdict"] = dict(panel_verdict)
        payload["judgment_panel_rule"] = (
            "The previous idea was measured and held, and a review panel still judged it short on "
            "this dimension. Do not defend it. Propose the idea the panel's advice points at.")
    if history:
        payload["lineage_history"] = history
    if must_change:
        payload["must_change_scientific_object"] = must_change
        payload["must_change_rule"] = (
            "Change this scientific object. A metaphor-only rewrite that keeps "
            "the same operationalization is an invalid escape."
        )
    if mistakes:
        payload["archive_mistakes_to_avoid"] = mistakes
    if operator in OPERATORS and not decision_request:
        payload["research_operator"] = {"name": operator, "definition": OPERATORS[operator]}
    if branch_kind:
        payload["branch_kind"] = branch_kind
    if stagnation:
        payload["why_the_lineage_is_stuck"] = stagnation
    instruction = MODE_INSTRUCTIONS[mode]
    if mode == REMOTE_BOOTSTRAP and (research_context or {}).get('exploration_environment'):
        instruction = ('Start from the user goal and supplied raw assets. Inspect them through '
                       'an exploratory program before choosing a research question or a transfer '
                       'mechanism. State a provisional question, not a completed research plan. '
                       'Do not infer what is in a file from its name alone.')
    if remote and remote.get("gap") and not remote.get("used"):
        instruction = (payload["remote_search_instruction"] +
                       " Preserve the user's problem and any constraints and feedback below.")
    if decision_request:
        payload["decision_request"] = decision_request
        instruction = (
            "Consider what the recorded evidence leaves unanswered. Choose whether the same question "
            "needs a new hypothesis, measurement, identification strategy, or a different question "
            "with an explicit question_transition. Explain the decision and its uncertainty. "
            "Keep the mission and hard constraints. Preserve the old hypothesis and experimental "
            "verdict. No repair operator is prescribed; you may name your chosen research_operator "
            "as a GENERATED planning judgment. Fill preserved_from_parent, changed_from_parent and addresses.")
    return instruction + "\n\n" + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def ideate(action: Any, state: Any, *, client: Any, workspace: Path | None,
           remote_search: Callable[..., Any] | None = None,
           projection: Mapping[str, Any] | None = None) -> CandidateResult:
    """One complete idea. Remote search runs only in the modes that may jump."""
    from ..llm import LLMUnavailable, complete_call

    view_state = readonly_view(state)
    extra = dict(getattr(action, "extra", {}) or {})
    mode = str(extra.get("mode") or REMOTE_BOOTSTRAP)
    if mode not in MODES:
        return _refused(f"unknown_discovery_mode:{mode}")
    if client is None:
        return _refused("research_model_not_connected")
    view = discovery_view(view_state)
    by_id = {str(row["id"]): row for row in view["ideas"]}
    parent = by_id.get(str(extra.get("parent_idea_id") or ""))
    if mode != REMOTE_BOOTSTRAP and parent is None:
        return _refused("parent_idea_missing")
    parent_fb = view["feedback"].get(str(parent["id"])) if parent else None
    decision_request = str(extra.get("decision_request") or "")
    references = extra.get('evidence_ids') or []
    from ..validity import evidence_eligible
    observed_ids = {r.get('evidence_id') for r in view_state.get('evidence_records', [])
        if parent and r.get('discovery_idea_id') == parent['id'] and evidence_eligible(r)
        and r.get('epistemic') == 'WORLD' and r.get('attested') and r.get('execution_status') == 'ran'}
    diagnosed_ids = {r.get('evidence_id') for r in view_state.get('evidence_records', [])
        if r.get('evidence_id') in observed_ids and r.get('diagnosed') and r.get('replay_ok') is True}
    evidence_decision = bool(mode == EVOLVE and decision_request
        and isinstance(references, (list, tuple)) and references
        and all(isinstance(eid, str) and eid in observed_ids for eid in references)
        and any(eid in diagnosed_ids for eid in references))
    # Verified observations can demand a research decision before a critic has
    # rated the proposal. A request string alone does not establish that basis.
    # A panel's verdict on a measured idea is feedback too. A registered local
    # experiment reaches its pilot without ever meeting the critic, so such an
    # idea has no `parent_fb`; the second live build scheduled the panel's
    # revision and refused it here twice, without a model call.
    panel_verdict: dict[str, Any] = {}
    if extra.get("judged_below_the_bar") and parent is not None:
        from .judgment import revision_request
        from .state import ScientificState as _State
        request = revision_request(_State.from_dict(dict(view_state)))
        if request and str(parent["id"]) in request.get("idea_ids", []):
            panel_verdict = {key: request[key] for key in (
                "dimension", "judged", "required", "decisive_argument", "what_would_raise_it")}
    from .exploratory import records as exploration_records, attempts as exploration_attempts
    exploratory_decision = bool(mode == EVOLVE and parent and
        (exploration_records(view_state, parent['id']) or exploration_attempts(view_state, parent['id'])))
    from .dependencies import dependency_view
    exploratory_decision = exploratory_decision or bool(mode == EVOLVE and parent and
        dependency_view(view_state).get(parent['id'], {}).get('status') == 'reconsideration_required')
    knowledge_decision = bool(mode == EVOLVE and parent and idea_of(parent).knowledge_request)
    if mode in (EVOLVE, REMOTE_ESCAPE) and not parent_fb and not evidence_decision and not panel_verdict and not exploratory_decision and not knowledge_decision:
        return _refused("evolution_requires_the_parent_feedback")
    branch_kind = str(extra.get("branch_kind") or "")
    operator = str(extra.get("operator") or "")
    if mode == REMOTE_BOOTSTRAP:
        operator = ""
    elif operator not in OPERATORS and not (mode == EVOLVE and decision_request):
        return _refused(f"unknown_research_operator:{operator}")
    prompt_operator = branch_kind if branch_kind in OPERATORS else operator
    from .state import ScientificState
    from .journal import research_journal
    research_context = research_journal(ScientificState.from_dict(dict(view_state)),
        theory_id=str(parent['id']) if parent else "")
    from .contract import contract_from_state
    research_context["mission_contract"] = contract_from_state(view_state).to_dict()
    from .requirement import world_capability_view
    from ..world import load_fixture
    research_context['bound_world_apis'] = []
    for world in view_state.get('world_versions', []):
        if world.get('deprecated') or not world.get('attested'):
            continue
        wid = str(world.get('id') or world.get('world_id') or '')
        location = world.get('path')
        if not location and workspace is not None and wid:
            from .experiment import confined_path
            try:
                location = confined_path(workspace, 'worlds', wid)
            except ValueError:
                continue
        if not location:
            continue
        try:
            fixture = load_fixture(Path(location))
            if fixture.id != wid or fixture.digest != world.get('digest'):
                continue
            interface = world_capability_view(fixture)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if interface['interfaces_attested']:
            research_context['bound_world_apis'].append(interface)
    research_context["research_questions"] = [dict(q) for key in (
        "open_questions", "resolved_questions", "partial_questions", "blocked_questions")
        for q in view_state.get(key, [])]
    research_context["live_blocker_references"] = [
        {"id": row.get("review_id"), "reason": row.get("objection")}
        for row in view_state.get("reviews", [])
        if row.get("status") in {"open", "repairing"}
        and (row.get("blocking") or row.get("severity") in {"major", "fatal"})]
    research_context["live_blocker_references"].extend(
        {"id": row.get("id"), "reason": row.get("blocked_reason")}
        for row in view_state.get("blocked_questions", [])
        if row.get("blocked_reason") and row.get("blocked_reason") != "retired_by_project_pivot")
    lineage_id = str(extra.get('lineage_id') or (parent or {}).get('lineage_id') or 'L1')
    policy = current_policy(view_state, lineage_id)
    from .memory import lineage_learning
    research_context['local_learning'] = lineage_learning(view_state, lineage_id=lineage_id,
        parent_id=str((parent or {}).get('id') or ''))
    from .resources import allocation_view
    research_context['budget_allocation'] = allocation_view(state, action)
    root = root_question(view_state)
    literature = _literature(view_state)
    from .inquiry import inquiry_context
    research_context['knowledge_inquiry'] = inquiry_context(view_state, parent, parent_fb)
    from .response import result_packet, validate_response
    research_context['result_packet'] = result_packet(view_state, str((parent or {}).get('id') or ''))
    # A rejected JSON response is an operational failure, not a failed idea.
    # The next paid attempt must see the concrete error it is meant to repair.
    research_context['recent_generation_failures'] = [
        {k: r.get(k) for k in ('status', 'reason', 'outcome_key')}
        for r in view_state.get('artifacts', [])
        if r.get('kind') == 'action_attempt' and r.get('action_type') == 'IDEATE'
        and r.get('target') == action.target and r.get('reason')
        and int(r.get('project_epoch') or 0) == int(view_state.get('project_epoch') or 0)
        and r.get('status') != 'service_unavailable'][-3:]
    remote: dict[str, Any] = {
        "attempted": False,
        "used": False,
        "status": "not_requested",
        "reason": ("idea evolution starts from the previous complete idea and its "
                   "validation feedback, not from a new latent point")}
    wants_remote = mode in (REMOTE_BOOTSTRAP, REMOTE_ESCAPE) or (
        mode == BRANCH_DIVERSIFICATION and branch_kind == REMOTE_BRANCH)
    if mode == REMOTE_BOOTSTRAP and research_context.get('exploration_environment'):
        wants_remote = False
        remote['reason'] = 'Inspect supplied assets before selecting a transfer bottleneck.'
    if wants_remote:
        if remote_search is None:
            remote = {"attempted": False, "used": False, "status": "unavailable",
                      "gap": "remote_search_unavailable",
                      "reason": "no remote search backend is connected"}
        else:
            used = [label for row in view["remote_searches"] for label in row["frames"] if label]
            intent = root if parent is None else f"{root} {idea_of(parent).central_claim}"
            try:
                raw = remote_search(purpose=mode, intent=intent,
                                    context=[w["title"] for w in literature["works"][:6]],
                                    exclude=used,
                                    k=int((policy.get("remote") or {}).get("frames_per_jump") or 3))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raw = {"backend": "error", "gap": f"remote_search_failed:{exc}"}
            remote = as_remote_result(raw, purpose=mode)
    prompt = ideation_prompt(
        mode=mode, root=root, operator=prompt_operator, branch_kind=branch_kind, parent=parent,
        parent_fb=parent_fb,
        history=_lineage_history(view, str(parent.get("lineage_id") or "")) if parent else None,
        mistakes=_archive_mistakes(view, lineage_id=lineage_id), remote=remote, literature=literature,
        stagnation=list(extra.get("stagnation") or []),
        must_change=str(extra.get("must_change") or ""), research_context=research_context,
        decision_request=decision_request, panel_verdict=panel_verdict)
    try:
        completion = complete_call(client, prompt, purpose=f"discovery_ideate_{mode.lower()}",
                                   system=ideation_system(research_context))
        payload = parse_json_object(getattr(completion, "text", ""))
    except LLMUnavailable as exc:
        return _service_unavailable(exc)
    except (ValueError, TypeError, KeyError) as exc:
        return _refused(f"idea_unparseable:{exc}", status="idea_unparseable")
    response = payload.get('result_response')
    response_gap = validate_response(research_context['result_packet'], response)
    if response_gap:
        return _refused(response_gap)
    if response and response['disposition'] == 'pause':
        return CandidateResult(status='idea_paused', events=[make_event('ArtifactAdded', {
            'kind': 'discovery_decision', 'parent_id': str((parent or {}).get('id') or ''),
            'project_epoch': int(view_state.get('project_epoch') or 0),
            'result_response': response, 'epistemic': 'GENERATED'}, action_type=IDEATE)])
    idea = CompleteIdea.from_payload(payload)
    if idea.exploration:
        from .exploratory import proposal_error
        if not proposal_error(idea.exploration):
            # Exploration can build the question and instrument. Missing
            # scientific commitments are explicitly absent, never invented to
            # make a preliminary program look like a complete hypothesis.
            idea = replace(idea,
                research_problem=idea.research_problem or idea.exploration['question'],
                central_claim=idea.central_claim or 'Provisional question: ' + idea.exploration['question'],
                mechanism=idea.mechanism or 'Not yet hypothesized; exploratory execution only.')
    if idea.transfer_analysis:
        from .inquiry import strings, text
        raw_transfer = idea.transfer_analysis
        available = {w['cite_id'] for w in literature['works']}
        available.update(str(s.get('cite_id')) for s in literature.get('trajectory_sources', []))
        cited = strings(raw_transfer.get('source_ids'))
        idea = replace(idea, transfer_analysis={
            **{k: text(raw_transfer.get(k)) for k in ('source_mechanism', 'target_mapping', 'discriminating_test',
                'implementation_correspondence', 'performance_test', 'mechanism_test')},
            **{k: strings(raw_transfer.get(k)) for k in ('required_assumptions', 'failure_conditions', 'rejected_frames', 'landing_questions')},
            'source_ids': [sid for sid in cited if sid in available],
            'unresolved_source_ids': [sid for sid in cited if sid not in available],
            'source_status': 'retrieved_identity_not_verified_transfer' if cited and set(cited) <= available
                             else 'requires_source_check', 'epistemic': 'GENERATED'})
    core = idea.missing_core()
    if core:
        # A pair of concepts, or a fragment, is not an idea. Nothing enters the
        # archive; the attempt itself is recorded by the controller.
        return _refused("complete_idea_core_fields_missing:" + ",".join(core),
                        status="idea_incomplete", missing=list(core))
    if not idea.grounding:
        idea = replace(idea, grounding=((
            "no verified literature was available; grounded only in the research intent"
            if not literature["works"] else "no verified work was cited"),))
    used_ids = {str(r.get('id') or '') for r in view_state.get('artifacts', [])
                if r.get('kind') == IDEA_KIND}
    sequence = len(used_ids) + 1
    while f"idea-{sequence:03d}" in used_ids:
        sequence += 1
    idea_id = f"idea-{sequence:03d}"
    hop = int(extra.get("hop") or (int(parent.get("hop") or 1) + 1 if parent else 1))
    row = {
        "kind": IDEA_KIND,
        "id": idea_id,
        "lineage_id": lineage_id,
        "hop": hop,
        "parent_id": str(parent["id"]) if parent else "",
        "question_id": str(extra.get("question_id") or (parent or {}).get("question_id") or ""),
        "mode": mode,
        "operator": operator,
        "operator_basis": str(extra.get("operator_basis") or ""),
        "branch_kind": branch_kind,
        "addresses_feedback": str((parent_fb or {}).get("id") or ""),
        "policy_version": str(policy.get("version") or ""),
        "idea": idea.to_dict(),
        "missing_fields": list(idea.missing()),
        "remote_search": remote,
        "literature": {"verified_works": len(literature["works"]),
                       "trajectory_hits": literature["transitions"]},
        "stagnation": list(extra.get("stagnation") or []),
        "model": str(getattr(completion, "model", "") or ""),
        "completion_digest": str(getattr(completion, "digest", "") or ""),
        "epistemic": "GENERATED",
        "project_epoch": int(view_state.get('project_epoch') or 0),
        "decision_evidence_ids": [r['evidence_id'] for r in
                                  research_context['relevant_world_evidence'] if r.get('evidence_id')],
    }
    if response is not None:
        row['result_response'] = response
    if "question_transition" in payload:
        # Keep the proposal inspectable. The admission boundary and reducer,
        # not the worker, decide whether its recorded references are valid.
        row["question_transition"] = payload["question_transition"]
    if decision_request:
        row["decision_request"] = decision_request
        row["proposed_research_operator"] = str(payload.get("research_operator") or "")
    row["files"] = _write_idea_files(workspace, row)
    return CandidateResult(
        status="ideated",
        events=[make_event("ArtifactAdded", row, action_type=IDEATE,
                           frontier_target_id=getattr(action, "frontier_target_id", "") or "goal:G0")],
        extras={"idea_id": idea_id, "mode": mode, "operator": operator,
                "remote_search_used": bool(remote.get("used")), "epistemic": "GENERATED"})


# ---------------------------------------------------------------------------
# CRITIQUE
# ---------------------------------------------------------------------------


def critique(action: Any, state: Any, *, client: Any, workspace: Path | None) -> CandidateResult:
    """V1 structural validation + literature grounding + V2 rollout → feedback packet."""
    from ..llm import LLMUnavailable, complete_call
    from .journal import research_journal
    from .state import ScientificState

    view_state = readonly_view(state)
    extra = dict(getattr(action, "extra", {}) or {})
    idea_id = str(extra.get("idea_id") or getattr(action, "target", "") or "")
    view = discovery_view(view_state)
    row = next((item for item in view["ideas"] if str(item["id"]) == idea_id), None)
    if row is None:
        return _refused("idea_missing")
    if idea_id in view["feedback"]:
        return _refused("idea_already_validated")
    if client is None:
        return _refused("research_model_not_connected")
    idea = idea_of(row)
    parent_row = next((item for item in view["ideas"]
                       if str(item.get("id") or "") == str(row.get("parent_id") or "")), None)
    structural = structural_validation(
        idea, remote_used=bool((row.get("remote_search") or {}).get("used")),
        operator=str(row.get("operator") or ""),
        parent=idea_of(parent_row) if parent_row is not None else None)
    grounding = literature_grounding(view_state, idea)
    chain = lineages(view["ideas"]).get(str(row.get("lineage_id") or "L1")) or []
    pairs = [(dict(item), dict(view["feedback"].get(str(item["id"])) or {}))
             for item in chain if int(item.get("hop") or 1) < int(row.get("hop") or 1)]
    payload = {
        "research_intent": root_question(view_state),
        "idea": idea.to_dict(),
        "idea_provenance": {"mode": row.get("mode"), "operator": row.get("operator"),
                            "hop": row.get("hop"),
                            "remote_frames": [f.get("label") for f in
                                              (row.get("remote_search") or {}).get("frames") or []]},
        "structural_findings": [item["detail"] for item in structural],
        "literature_grounding": {key: grounding.get(key) for key in
                                 ("status", "verified_works", "detail")},
        "lineage_history": _lineage_history(view, str(row.get("lineage_id") or "L1"))[:-1],
        "evidence_route": evidence_route(idea, view_state),
        "observed_research_context": research_journal(ScientificState.from_dict(view_state)),
        "source_context": _literature(view_state),
    }
    import os
    from .resources import compact_critique

    ledger = getattr(client, 'ledger', None)
    remaining_calls = ledger.remaining().get('api_calls') if callable(getattr(ledger, 'remaining', None)) else None
    fast = os.environ.get("FARFIELD_CRITIQUE_FAST") == "1" or compact_critique(remaining_calls)

    try:
        packed = json.dumps(payload, ensure_ascii=False, indent=1)
        if fast:
            completion = complete_call(client, packed, purpose="discovery_critique",
                                       system=SYSTEM_CRITIQUE_FAST)
            answer = parse_json_object(getattr(completion, "text", ""))
            rollout = normalize_rollout(answer.get("rollout") or answer.get("research_world_rollout"))
            critique_body = answer.get("critique") or {}
            digest = str(getattr(completion, "digest", "") or "")
            model_name = str(getattr(completion, "model", "") or "")
        else:
            rolled = complete_call(client, packed, purpose="discovery_rollout",
                                   system=SYSTEM_ROLLOUT)
            rollout = normalize_rollout(parse_json_object(getattr(rolled, "text", "")))
            critic = complete_call(
                client,
                json.dumps({**payload, "research_world_rollout": rollout},
                           ensure_ascii=False, indent=1),
                purpose="discovery_critique",
                system=SYSTEM_CRITIQUE,
            )
            critique_body = parse_json_object(getattr(critic, "text", ""))
            if "critique" in critique_body and isinstance(critique_body["critique"], Mapping):
                critique_body = critique_body["critique"]
            digest = str(getattr(critic, "digest", "") or "")
            model_name = str(getattr(critic, "model", "") or "")
    except LLMUnavailable as exc:
        return _service_unavailable(exc)
    except (ValueError, TypeError, KeyError) as exc:
        return _refused(f"critique_unparseable:{exc}", status="critique_unparseable")
    feedback = compose_feedback(idea_row=row, structural=structural, grounding=grounding,
                                rollout=rollout, critique=critique_body,
                                policy=current_policy(view_state, str(row.get('lineage_id') or 'L1')),
                                pairs=pairs, state=view_state)
    feedback["model"] = model_name
    feedback["completion_digest"] = digest
    feedback["rollout_separate_from_critique"] = not fast
    feedback["budget_execution"] = {'mode': 'joint' if fast else 'separate',
        'remaining_calls_at_start': remaining_calls,
        'reason': 'preserve_revision_capacity' if compact_critique(remaining_calls) else 'configured_roles',
        'quality_equivalence_verified': False}
    feedback["files"] = _write_feedback_files(workspace, row, feedback, view_state)
    return CandidateResult(
        status=f"critiqued_{feedback['verdict'].lower()}",
        events=[make_event("ArtifactAdded", feedback, action_type=CRITIQUE,
                           frontier_target_id=getattr(action, "frontier_target_id", "") or "goal:G0")],
        extras={"idea_id": idea_id, "verdict": feedback["verdict"],
                "root_failure_cause": feedback["root_failure_cause"],
                "recommended_next_operator": feedback["recommended_next_operator"],
                "epistemic": "GENERATED"})


# ---------------------------------------------------------------------------
# Research artifacts: IDEA.md, RESEARCH_PLAN.md, EXPERIMENT_DESIGN.md
# ---------------------------------------------------------------------------

_BANNER = ("> GENERATED by FarField's discovery runtime. This is a research proposal, not "
           "evidence. Research promise and evidence status are tracked separately: an idea can "
           "be worth pursuing before it is empirically proven.\n")


def _lines(items: Any, *, empty: str = "—") -> str:
    rows = [str(item) for item in (items or []) if str(item).strip()]
    return "\n".join(f"- {item}" for item in rows) if rows else empty


def render_idea(row: Mapping[str, Any]) -> str:
    idea = idea_of(row)
    remote = dict(row.get("remote_search") or {})
    parts = [
        f"# IDEA — {idea.title or idea.central_claim[:80]}\n",
        _BANNER,
        f"- Idea: `{row.get('id')}` · lineage `{row.get('lineage_id')}` · hop {row.get('hop')}",
        f"- Mode: {row.get('mode')}" + (f" · operator {row.get('operator')}" if row.get("operator") else "")
        + (f" · branch {row.get('branch_kind')}" if row.get("branch_kind") else ""),
        f"- Parent: `{row.get('parent_id') or '—'}` · addresses feedback "
        f"`{row.get('addresses_feedback') or '—'}`",
        f"- Exploration policy: {row.get('policy_version') or '—'} · claim type: "
        f"{idea.claim_type or 'unstated (routed as conceptual)'}",
        "",
    ]
    for title, value in (
        ("Research problem", idea.research_problem), ("Central claim", idea.central_claim),
        ("Mechanism", idea.mechanism), ("Motivation", idea.motivation),
        ("Why it may matter", idea.why_it_may_matter),
        ("Novelty hypothesis", idea.novelty_hypothesis),
        ("Strongest competing explanation", idea.competing_explanation),
        ("Falsifier", idea.falsifier), ("Validation direction", idea.validation_direction),
        ("Optional experiment direction", idea.optional_experiment_direction),
    ):
        parts.append(f"## {title}\n\n{value or '—'}\n")
    parts.append(f"## Grounding\n\n{_lines(idea.grounding)}\n")
    parts.append(f"## Uncertainties\n\n{_lines(idea.uncertainties)}\n")
    if idea.structural_correspondence:
        parts.append(f"## Structural correspondence\n\n{idea.structural_correspondence}\n")
    if idea.transfer_analysis:
        parts.append("## Transfer analysis (GENERATED)\n\n```json\n" +
                     json.dumps(idea.transfer_analysis, ensure_ascii=False, indent=2) + "\n```\n")
    if row.get("parent_id"):
        parts.append(f"## Evolution from the parent idea\n\n**Preserved**\n\n"
                     f"{_lines(idea.preserved_from_parent)}\n\n**Changed**\n\n"
                     f"{_lines(idea.changed_from_parent)}\n\n**Addresses**\n\n"
                     f"{idea.addresses or '—'}\n")
    if remote.get("used"):
        basin = [r.get("label") for r in remote.get("local_basin") or [] if isinstance(r, Mapping)]
        frames = [f"{f.get('label')} (distance {f.get('distance')})" for f in remote.get("frames") or []]
        parts.append("## Remote search provenance\n\nLocal basin:\n\n" + _lines(basin)
                     + "\n\nRemote frames offered:\n\n" + _lines(frames)
                     + "\n\nDistance is provenance about where a frame came from, never a "
                       "measure of the idea's quality.\n")
    else:
        parts.append(f"## Remote search\n\nNot used: {remote.get('reason') or remote.get('gap') or '—'}\n")
    if row.get("missing_fields"):
        parts.append(f"## Incomplete fields\n\n{_lines(row.get('missing_fields'))}\n")
    response = row.get('result_response')
    if response:
        parts.append('## Response to recorded results\n\n'
                     + str(response['disposition']) + ': ' + str(response['rationale'])
                     + '\n\nNext step: ' + str(response['next_step']) + '\n\n'
                     + _lines([r['result_id'] + ': ' + r['interpretation']
                               for r in response['addressed_results']]) + '\n')
    return "\n".join(parts)


def render_plan(row: Mapping[str, Any], fb: Mapping[str, Any], state: Any) -> str:
    from .contract import contract_from_state
    from .quality import requirements_for

    idea = idea_of(row)
    investment = dict(fb.get('investment') or {})
    next_check = dict(investment.get('next_check') or {})
    from .inquiry import inquiry_context
    inquiry = inquiry_context(state, row, fb)
    rollout = dict(fb.get("rollout") or {})
    requirements = requirements_for(contract_from_state(state)) if hasattr(state, "get") else {}
    if fb.get("verdict") == PROMISING:
        next_move = ("Candidate for deeper validation. "
                     + ("Next: compile it into a registered, world-bound hypothesis for the "
                        "evidence runtime." if fb.get("evidence_route") == "V4_EVIDENCE_RUNTIME"
                        else "Its claim still requires a checked argument and source evidence; "
                             "the evidence runtime's two-arm test does not apply."
                        if fb.get("evidence_route") == "NOT_APPLICABLE"
                        else "Its empirical test waits for an attested world.")
                     + (" An alternative lineage may branch from it."
                        if fb.get("recommended_next_operator") == BRANCH_DIVERSIFICATION else ""))
    else:
        next_move = (f"Revise with {fb.get('recommended_next_operator') or '—'} "
                     f"({fb.get('operator_basis') or ''}).\n\nPreserve:\n\n"
                     f"{_lines(fb.get('preserve_constraints') or fb.get('what_survives'))}\n\n"
                     f"Change:\n\n{_lines(fb.get('change_constraints'))}\n\nAvoid:\n\n"
                     f"{_lines(fb.get('mistakes_to_avoid'))}")
    return "\n".join([
        f"# RESEARCH PLAN — {idea.title or idea.central_claim[:80]}\n",
        _BANNER,
        f"- Idea `{row.get('id')}` · verdict **{fb.get('verdict')}** · research promise "
        f"**{fb.get('research_promise')}** · evidence route {fb.get('evidence_route')}",
        f"- Root failure cause: {fb.get('root_failure_cause')} · research-world outcome: "
        f"{rollout.get('outcome')} — {rollout.get('reason')}",
        "",
        f"## What survives\n\n{_lines(fb.get('what_survives'))}\n",
        f"## What failed\n\n{_lines(fb.get('what_failed'))}\n",
        f"## Strongest competing explanation\n\n{fb.get('strongest_competing_explanation') or '—'}\n",
        f"## Likely research path\n\n{_lines(rollout.get('possible_research_path'))}\n",
        f"## Dependencies\n\n{_lines(rollout.get('dependency_structure'))}\n",
        f"## Next questions\n\n{_lines(fb.get('new_questions'))}\n",
        f"## Required evidence\n\n{_lines(fb.get('required_evidence'))}\n",
        f"## Kill criteria\n\n{_lines([rollout.get('what_would_kill_the_idea'), idea.falsifier])}\n",
        f"## What would strengthen it\n\n{rollout.get('what_result_would_strengthen_it') or '—'}\n",
        "## Next research investment (provisional)\n\n"
        f"- Value if the assumption holds: {investment.get('conditional_value') or 'unspecified'}\n"
        f"- Discriminating check: {investment.get('discriminating_check') or 'unspecified'}\n"
        f"- Question to resolve: {next_check.get('question') or 'unspecified'}\n"
        f"- Knowledge preparation: {inquiry.get('status') or 'no literature inquiry proposed'}\n"
        f"- Cost: {investment.get('cost_status') or 'not measured'}\n"
        f"- Stop rule: {investment.get('stop_rule') or 'unspecified'}\n\n"
        "Missing budget or sources means under-assessed, not refuted.\n",
        f"## Next move\n\n{next_move}\n",
        "## Publication-grade requirements (not gates on this idea)\n\n"
        + _lines([f"{name}: {level}" for name, level in sorted(requirements.items())])
        + "\n\nThese are what the evidence runtime will demand before any claim is called "
          "supported. An early idea is not invalid for lacking them.\n",
    ])


def render_design(row: Mapping[str, Any], fb: Mapping[str, Any]) -> str:
    idea = idea_of(row)
    rollout = dict(fb.get("rollout") or {})
    route = str(fb.get("evidence_route") or "")
    route_text = {
        "V4_EVIDENCE_RUNTIME": ("Empirical claim on a bound, attested world: the evidence runtime "
                                "compiles it into a registered hypothesis and runs a pre-registered "
                                "two-arm test."),
        "AWAITING_ATTESTED_WORLD": ("Empirical claim with no attested world bound: the design is "
                                    "ready, the experiment cannot run yet."),
        "NOT_APPLICABLE": ("Conceptual or theoretical claim: requires a checked argument and "
                           "source evidence. The two-arm freeze test does not apply."),
    }.get(route, route)
    from .requirement import compile_experiment_requirement

    req = compile_experiment_requirement(idea, idea_id=str(row.get("id") or ""), feedback=fb)
    return "\n".join([
        f"# EXPERIMENT DESIGN — {idea.title or idea.central_claim[:80]}\n",
        _BANNER + "> Nothing in this design has been run.\n",
        "## Hypotheses compared\n",
        f"- **H (idea `{row.get('id')}`)**: {idea.central_claim}",
        f"- **A (strongest competing explanation)**: "
        f"{fb.get('strongest_competing_explanation') or idea.competing_explanation or '—'}\n",
        f"## Discriminating outcome (falsifier)\n\n{idea.falsifier or '—'}\n",
        f"## Validation direction\n\n{idea.validation_direction or '—'}\n",
        f"## Proposed experiment\n\n{idea.optional_experiment_direction or '—'}\n",
        "## Compiled experiment requirement (GENERATED)\n\n"
        f"- world class: `{req.world_class or 'unsupported'}`\n"
        f"- primary metric: `{req.primary_metric or '—'}`\n"
        f"- intervention: {req.intervention or '—'}\n"
        f"- control: {req.control_condition or '—'}\n"
        f"- falsifier: {req.falsifier or '—'}\n"
        f"- required operations: {', '.join(req.required_operations) or '—'}\n",
        f"## Evidence route\n\n{route_text}\n",
        "## What evidence promotion will require\n\n"
        "- both arms measured with the same registered measure, direction and margin fixed before "
        "either runs\n- an oracle placebo where the measure depends on an oracle\n- placebo worlds "
        "(at least five) with the result beyond every one of them\n- the sample each arm measured, "
        "reported by the experiment\n- the competing mechanism ablated\n- a replication on a heldout "
        "world\n\nThese are the trusted kernel's requirements, restated. This document does not "
        "satisfy any of them.\n",
        f"## What would kill it\n\n{rollout.get('what_would_kill_the_idea') or '—'}\n",
        f"## What would strengthen it\n\n{rollout.get('what_result_would_strengthen_it') or '—'}\n",
    ])


def _idea_dir(workspace: Path | None, idea_id: str) -> Path | None:
    if workspace is None:
        return None
    folder = Path(workspace) / DISCOVERY_DIR / "ideas" / str(idea_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_idea_files(workspace: Path | None, row: Mapping[str, Any]) -> dict[str, str]:
    folder = _idea_dir(workspace, str(row.get("id") or ""))
    if folder is None:
        return {}
    (folder / "IDEA.md").write_text(render_idea(row), encoding="utf-8")
    (folder / "idea.json").write_text(json.dumps(dict(row), ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    return {"idea_md": str(folder / "IDEA.md"), "idea_json": str(folder / "idea.json")}


def _write_feedback_files(workspace: Path | None, row: Mapping[str, Any],
                          fb: Mapping[str, Any], state: Any) -> dict[str, str]:
    folder = _idea_dir(workspace, str(row.get("id") or ""))
    if folder is None:
        return {}
    (folder / "FEEDBACK.json").write_text(json.dumps(dict(fb), ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    (folder / "ROLLOUT.json").write_text(
        json.dumps(dict(fb.get("rollout") or {}), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (folder / "RESEARCH_PLAN.md").write_text(render_plan(row, fb, state), encoding="utf-8")
    (folder / "EXPERIMENT_DESIGN.md").write_text(render_design(row, fb), encoding="utf-8")
    return {name: str(folder / name) for name in
            ("FEEDBACK.json", "ROLLOUT.json", "RESEARCH_PLAN.md", "EXPERIMENT_DESIGN.md")}


def write_deliverables(workspace: Path | None, state: Any) -> dict[str, str]:
    """The run's deliverables, for the selected idea, plus the discovery record.

    Rendered from the event-sourced artifacts, so they are reproducible from the
    log. The selected idea is the best research hypothesis by feedback, never
    the farthest one.
    """
    if workspace is None:
        return {}
    view = discovery_view(state)
    if not view["ideas"]:
        return {}
    root = Path(workspace)
    by_id = {str(row["id"]): row for row in view["ideas"]}
    selected = by_id[view["selected_idea"]]
    fb = view["feedback"].get(str(selected["id"]))
    written = {"IDEA.md": root / "IDEA.md"}
    written["IDEA.md"].write_text(render_idea(selected), encoding="utf-8")
    if fb:
        written["RESEARCH_PLAN.md"] = root / "RESEARCH_PLAN.md"
        written["RESEARCH_PLAN.md"].write_text(render_plan(selected, fb, state), encoding="utf-8")
        written["EXPERIMENT_DESIGN.md"] = root / "EXPERIMENT_DESIGN.md"
        written["EXPERIMENT_DESIGN.md"].write_text(render_design(selected, fb), encoding="utf-8")
    summary = discovery_summary(state)
    table = ["| Idea | Lineage | Hop | Mode | Operator | Remote search | Verdict | Root cause | "
             "Promise | Evidence |", "|---|---|---|---|---|---|---|---|---|---|"]
    for row in view["ideas"]:
        idea_fb = view["feedback"].get(str(row["id"])) or {}
        status = view["statuses"].get(str(row["id"])) or {}
        table.append(
            f"| `{row['id']}` | {row.get('lineage_id')} | {row.get('hop')} | {row.get('mode')} | "
            f"{row.get('operator') or '—'} | "
            f"{'yes' if (row.get('remote_search') or {}).get('used') else 'no'} | "
            f"{idea_fb.get('verdict', 'unvalidated')} | {idea_fb.get('root_failure_cause', '—')} | "
            f"{status.get('research_promise')} | {status.get('evidence_status')} |")
    policies = [f"{row.get('old_policy_version')} → {row.get('new_policy_version') or 'rejected'}: "
                f"{row.get('reason')} (used by {row.get('used_by_next_hop') or 'no hop yet'})"
                for row in view["policy_history"]]
    text = "\n".join([
        f"# DISCOVERY — {root_question(state)}\n",
        _BANNER,
        f"**{summary['headline']}**\n",
        "\n".join(summary["narrative"] + summary["closing"]) + "\n",
        f"Selected idea: `{summary['selected_idea']}` · research promise "
        f"{summary['research_promise']} · evidence status {summary['evidence_status']}\n",
        "## Ideas\n",
        "\n".join(table) + "\n",
        f"## Exploration policy\n\nCurrent: {view['policy_version']}\n\n{_lines(policies)}\n",
        "## Remote searches\n\n" + _lines(["%s (%s): %s" % (row["idea_id"], row["purpose"], row["frames"])
                                           for row in view["remote_searches"]]) + "\n",
    ])
    written["DISCOVERY.md"] = root / "DISCOVERY.md"
    written["DISCOVERY.md"].write_text(text, encoding="utf-8")
    lineage = root / DISCOVERY_DIR / "LINEAGE.json"
    lineage.parent.mkdir(parents=True, exist_ok=True)
    lineage.write_text(json.dumps({
        "lineages": view["lineages"], "statuses": view["statuses"],
        "policy_history": view["policy_history"], "selected_idea": view["selected_idea"],
        "epistemic": "GENERATED"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written["LINEAGE.json"] = lineage
    return {name: str(path) for name, path in written.items()}
