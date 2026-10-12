"""Termination semantics: seven different endings that were all one ending.

The old runtime had exactly two vocabularies for stopping — a ``should_terminate``
reason string (``budget`` / ``frontier_exhausted`` / ``stagnation``) and a final
report hard-coded to ``research_complete: False``. That is honest but useless:
it cannot tell "we answered the question" from "we ran out of money" from "this
direction cannot be answered with the worlds we have", and a controller that
cannot tell those apart cannot choose differently between them.

None of the following, alone or together, means the research is ready:

    the runtime loop finished normally
    every currently open question is resolved
    a complete paper was generated
    the budget ran out
    the frontier is momentarily empty

Readiness is a conjunction of gates over admitted evidence, recorded
objections, the contract, and the claim's actual scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contract import contract_from_state
from .quality import STATUS_RANK, meets, open_reviews, requirements_for, result_review_scope


RUN_SEGMENT_DONE = "RUN_SEGMENT_DONE"
RESEARCH_READY = "RESEARCH_READY"
RESEARCH_INCOMPLETE = "RESEARCH_INCOMPLETE"
BLOCKED = "BLOCKED"
NO_VIABLE_PATH = "NO_VIABLE_PATH"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
FRONTIER_EXHAUSTED = "FRONTIER_EXHAUSTED"
# The model service refused further calls in this session. The run stopped; no
# question was found unanswerable, and nothing about the science changed.
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
# The experiment ran, the reading is admitted, and there is no structured
# mission for it to answer. Four live missions measured exactly what was asked,
# confirmed it, and then spent twenty further ticks being advised to redesign
# experiments that were correct, because nothing recognised this as an ending.
# The science is not stuck; the question was never stated in a form any
# measurement can answer, and only the caller can supply that.
MISSION_NOT_DECLARED = "MISSION_NOT_DECLARED"
# The question is answered and a panel judged the answer short of the contract on
# novelty or importance, with the allowed revisions spent. Two live missions got
# that verdict — "an elementary consequence of the handshaking lemma" — and then
# spent fourteen further ticks redesigning an experiment nobody had faulted. No
# measurement can repair a judgment, so there is nothing left to run.
JUDGED_BELOW_CONTRACT = "JUDGED_BELOW_CONTRACT"

STATUSES = (
    RUN_SEGMENT_DONE,
    RESEARCH_READY,
    RESEARCH_INCOMPLETE,
    BLOCKED,
    NO_VIABLE_PATH,
    BUDGET_EXHAUSTED,
    FRONTIER_EXHAUSTED,
    SERVICE_UNAVAILABLE,
    MISSION_NOT_DECLARED,
    JUDGED_BELOW_CONTRACT,
)


@dataclass(frozen=True)
class Readiness:
    status: str
    gates: tuple[tuple[str, bool, str], ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == RESEARCH_READY

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(name for name, ok, _ in self.gates if not ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.ready,
            "publication_ready": False,
            "readiness_scope": "declared_contract_only_not_external_scientific_acceptance",
            "gates": [
                {"gate": name, "ok": bool(ok), "detail": detail}
                for name, ok, detail in self.gates
            ],
            "failed_gates": list(self.failed_gates),
            "reasons": list(self.reasons),
            "note": (
                "A finished run, a full paper, an empty frontier and an empty "
                "budget are all compatible with unfinished research."
            ),
        }


def _dimension(state: Any, name: str) -> dict[str, Any]:
    return dict(((state.quality or {}).get("dimensions") or {}).get(name) or {})


def _status(state: Any, name: str) -> str:
    return str(_dimension(state, name).get("status") or "unknown")


def readiness(state: Any, *, no_executable_action: bool = False) -> Readiness:
    """Is this project actually done, and if not, in what way is it not done?

    ``no_executable_action`` is the controller reporting that nothing in the
    frontier can currently run. That is a blockage to report, not a result.
    """
    contract = contract_from_state(state)
    quality = state.quality or {}
    requirements = requirements_for(contract)
    reviews = list(getattr(state, "reviews", []) or [])
    fatal = open_reviews(reviews, severity="fatal")
    major = open_reviews(reviews, severity="major")
    resources = dict(getattr(state, "resources", {}) or {})
    allowed_major = int((contract.quality_requirements or {}).get("allowed_open_major") or 0)

    # A gate over an empty set is not satisfied; it is unasked. "No row carries
    # a validity gap" said nothing when there were no rows. A row counts by what
    # its arms read before validity was applied: a separation that read as
    # support and carries a gap is exactly what this gate exists to catch, even
    # though the contract has since reduced its verdict to uninformative.
    def reading(row: Mapping[str, Any]) -> str:
        validity = row.get("validity") if isinstance(row.get("validity"), Mapping) else {}
        return str(validity.get("arithmetic_verdict") or row.get("outcome") or "")

    from ..validity import evidence_eligible
    retired_ids = {str(t.get("id") or "") for t in getattr(state, "theories", [])
                   if t.get("status") == "retired"}
    directional = [
        row for row in getattr(state, "evidence_records", []) or []
        if evidence_eligible(row)
        and str(row.get("theory_id") or "") not in retired_ids
        and row.get("study_stage") not in {"pilot", "calibration", "scientific_pilot"}
        and reading(row) in {"supports", "weakens"}
    ]
    out_of_scope = [row for row in directional if row.get("cannot_corroborate")]
    replication_required = requirements.get("replication", "missing")
    outputs = {str(row.get("kind") or "") for row in getattr(state, "artifacts", []) or []}
    missing_outputs = [name for name in contract.required_outputs if name not in outputs]
    answer_required = (contract.research_mode == 'open_research'
                       or 'central_question_answered' in contract.termination_requirements)
    answer = quality.get('question_answer') or {}

    gates: list[tuple[str, bool, str]] = [
        (
            "central_question_answered" if answer_required else "central_question_supported",
            meets(answer.get('strength', 'missing') if answer_required else _status(state, "central_claim_support"),
                  requirements.get("central_claim_support", "adequate")),
            (f"question_answer={answer.get('status', 'unresolved')}" if answer_required
             else f"central_claim_support={_status(state, 'central_claim_support')}"),
        ),
        ("no_fatal_objections", not fatal,
         f"{len(fatal)} open fatal objections"),
        ("major_objections_within_contract", len(major) <= allowed_major,
         f"{len(major)} open major objections, contract allows {allowed_major}"),
        (
            "required_comparisons_done",
            meets(_status(state, "baseline_strength"), requirements.get("baseline_strength", "partial")),
            f"baseline_strength={_status(state, 'baseline_strength')}",
        ),
        (
            "required_replication_done",
            meets(_status(state, "replication"), replication_required),
            f"replication={_status(state, 'replication')}; the contract requires "
            f"{replication_required}",
        ),
        ("evidence_scope_matches_claim", bool(directional) and not out_of_scope,
         (f"{len(out_of_scope)} of {len(directional)} directional WORLD rows carry "
          "validity gaps") if directional else "no directional WORLD evidence to scope a claim"),
        ("no_unauthorized_goal_drift",
         int((getattr(state, "debt", {}) or {}).get("goal_drift_debt") or 0) == 0,
         f"{int((getattr(state, 'debt', {}) or {}).get('goal_drift_debt') or 0)} drifted questions"),
        ("no_critical_unresolved_contradiction",
         not list(getattr(state, "contradictions", []) or []),
         f"{len(list(getattr(state, 'contradictions', []) or []))} open contradictions"),
        ("quality_gap_closed", not list(quality.get("quality_gap") or []),
         f"short on: {sorted(quality.get('quality_gap') or [])}"),
        ("required_outputs_present", not missing_outputs,
         f"missing outputs: {missing_outputs}"),
    ]
    mission_gaps = (quality.get('mission_completion') or {}).get('remaining_mission_gaps') or []
    if mission_gaps:
        gates.append(('mission_obligations_remain', False,
                      'derived questions leave unanswered mission obligations: ' + str(mission_gaps)))
    if ('result_review_current' in contract.termination_requirements
            or 'surviving review' in contract.success_target.casefold()):
        scope = result_review_scope(state)
        gates.append(('result_review_current', bool(scope['evidence_ids']) and any(
            r.get('result_review_scope') == scope['digest'] for r in reviews),
            'result review must cover the current evidence and scope'))
    if all(ok for _, ok, _ in gates):
        return Readiness(status=RESEARCH_READY, gates=tuple(gates))

    reasons = tuple(detail for name, ok, detail in gates if not ok)

    # A spent service balance stops the run before anything else is concluded
    # about it; reporting it as BLOCKED said a question could not be answered.
    services = list(resources.get("service_exhausted") or [])
    if services:
        return Readiness(status=SERVICE_UNAVAILABLE, gates=tuple(gates),
                         reasons=reasons + (f"model service refused further calls: {services}",))

    # Budget exhaustion settles the *run*, never the science. It is reported as
    # its own status so nobody reads it as an answer.
    if resources.get("exhausted"):
        return Readiness(status=BUDGET_EXHAUSTED, gates=tuple(gates), reasons=reasons)

    # No viable path: the objection cannot be repaired inside this project, or
    # the question cannot be asked of any world we have.
    pivot_demanded = [row for row in open_reviews(reviews)
                      if str(row.get("repair_type") or "") == "project_pivot"]
    revisions = [row for row in getattr(state, "design_revisions", []) or []
                 if int(row.get("project_epoch") or 0) == int(getattr(state, "project_epoch", 0))]
    exhausted_repairs = len(revisions) >= 3 and all(
        str(row.get("outcome") or "open") != "resolved" for row in revisions)
    if pivot_demanded and exhausted_repairs:
        return Readiness(status=NO_VIABLE_PATH, gates=tuple(gates),
                         reasons=reasons + ("redesign_alternatives_exhausted",))

    # An admitted, corroborating measurement with no structured mission to
    # answer is an ending, not a frontier. Reported before the capability
    # blockages below because nothing here is missing except the declaration.
    # ...but not while registered measurement work is still open. A live run
    # stopped a five-setting study after its first setting because the mission
    # was undeclared, leaving four pre-registered, zero-cost readings untaken.
    # What was registered gets measured; the declaration question can wait for
    # a complete record to attach to.
    coverage_open = any(
        row.get("kind") == "coverage_gap"
        for row in (getattr(state, "frontier", {}) or {}).get("measurement_gaps") or ())
    if (answer.get('unresolved_reason') == 'mission_has_no_structured_definition'
            and not coverage_open
            and any(row.get('study_stage') in {'confirmatory', 'replication'}
                    for row in directional)):
        return Readiness(status=MISSION_NOT_DECLARED, gates=tuple(gates),
                         reasons=reasons + (str(answer.get('unlock') or
                             'declare the mission as a registered comparison'),))

    from .judgment import judged_short, revision_request
    from .quality import JUDGMENT_DIMENSIONS
    short = judged_short(state)
    open_gaps = set((getattr(state, "quality", {}) or {}).get("quality_gap") or [])
    if (short and open_gaps and open_gaps <= set(JUDGMENT_DIMENSIONS) and open_gaps <= set(short)
            and not revision_request(state)):
        return Readiness(status=JUDGED_BELOW_CONTRACT, gates=tuple(gates), reasons=reasons + tuple(
            f"{name} judged {row['judged']}, contract requires {row['required']}: "
            f"{row['what_would_raise_it'] or row['decisive_argument']}"
            for name, row in sorted(short.items())))

    # What the project could not measure, in the words of the proposal and of
    # the instruments. A blocked run that can only say "not operationalised"
    # leaves whoever reads it to rediscover the reason.
    instruments = []
    for row in (getattr(state, "frontier", {}) or {}).get("measurement_gaps") or ():
        gap = row.get("instrument")
        if not isinstance(gap, Mapping):
            continue
        wanted = (str(gap["requested"].get("primary_metric") or "")
                  or ", ".join(gap.get("missing_observables") or ())
                  or "the requested observable")
        available = gap["available"]
        if not available.get("world_class_supported", True):
            instruments.append(
                f"{row.get('idea_id') or 'a proposal'} asks to measure {wanted} on a "
                f"{available.get('world_class') or 'unnamed'} world; this project has no "
                f"instrument for that kind of object, only for "
                f"{', '.join(available.get('known_world_classes') or ())}")
            continue
        offered = ", ".join(available.get("observables") or ()) or "nothing"
        instruments.append(
            f"{row.get('idea_id') or 'a proposal'} asks to measure {wanted}; "
            f"the available instruments read {offered}")
    if instruments:
        reasons = reasons + (f"instrument gap: {instruments[0]}",)

    blocked = list(getattr(state, "blocked_questions", []) or [])
    missing = list(getattr(state, "missing_capabilities", []) or [])
    if blocked and not list(getattr(state, "open_questions", []) or []) and missing:
        return Readiness(status=BLOCKED, gates=tuple(gates),
                         reasons=reasons + (f"missing capabilities: {missing}",))
    if no_executable_action and (blocked or missing or
                                 list(getattr(state, "open_questions", []) or [])):
        # Unanswered questions and nothing that can be run against them: the
        # project is waiting on capability, not finished and not out of ideas.
        detail = f"missing capabilities: {missing}" if missing else (
            "no eligible action can execute against the open frontier")
        return Readiness(status=BLOCKED, gates=tuple(gates), reasons=reasons + (detail,))

    if not (getattr(state, "open_questions", None) or getattr(state, "blocked_questions", None)):
        return Readiness(status=FRONTIER_EXHAUSTED, gates=tuple(gates),
                         reasons=reasons + ("frontier is empty; that is not an answer",))

    return Readiness(status=RESEARCH_INCOMPLETE, gates=tuple(gates), reasons=reasons)


def run_outcome(state: Any, *, horizon_reached: bool = False,
                no_executable_action: bool = False) -> dict[str, Any]:
    """The report a caller should actually read at the end of a segment."""
    verdict = readiness(state, no_executable_action=no_executable_action)
    payload = verdict.to_dict()
    payload["research_complete"] = verdict.ready
    payload['task_delivery'] = task_delivery(state)
    payload['task_complete'] = payload['task_delivery']['complete']
    payload['scientific_confirmation'] = {'complete': verdict.ready, 'status': verdict.status,
        'scope': 'existing_scientific_contract_gates_not_native_submission_scores'}
    if horizon_reached and not verdict.ready:
        # A segment ending is not a research ending; say both.
        payload["segment"] = RUN_SEGMENT_DONE
    return payload


def task_delivery(state: Any) -> dict[str, Any]:
    """Host-frozen deliverable acceptance, separate from scientific confirmation.

    Empty metric limits mean a valid evaluated submission, including a zero
    score. They do not mean a successful method. A threshold must be frozen by
    the task provider if achieving a performance target is required for delivery.
    """
    from .dependencies import dependency_view
    artifacts = getattr(state, 'artifacts', []) or []
    contract = next((r for r in reversed(artifacts)
                     if r.get('kind') == 'native_evaluation_environment'), {})
    policy = contract.get('delivery_policy')
    result = {'complete': False, 'status': 'not_configured' if policy is None else 'pending',
              'stop_on_acceptance': bool((policy or {}).get('stop_on_acceptance')),
              'scientific_confirmation_implied': False}
    rows = [r for r in artifacts if r.get('kind') == 'native_evaluation'
            and r.get('project_epoch', 0) == getattr(state, 'project_epoch', 0)]
    if not rows:
        return result
    latest = rows[-1]  # Never hide a failed last submission behind a previous best.
    result.update(evaluation_id=latest['id'], evaluation_status=latest['status'])
    current = dependency_view(state).get(latest['id'], {}).get('status') == 'current'
    valid = latest['status'] == 'evaluated' and latest.get('contract_digest') == contract.get('contract_digest')
    result['submission_valid'] = valid and current
    if policy is None:
        return result
    if not current or not valid:
        result['status'] = 'invalidated' if not current else 'evaluation_failed'
        return result
    metrics = latest.get('metrics', {})
    failed = [name for name, value in policy.get('minimum_metrics', {}).items()
              if name not in metrics or metrics[name] < value]
    failed += [name for name, value in policy.get('maximum_metrics', {}).items()
               if name not in metrics or metrics[name] > value]
    result.update(complete=not failed, status='below_target' if failed else 'complete',
                  unmet_metrics=sorted(set(failed)), acceptance_basis='host_frozen_delivery_policy')
    return result
