"""How many times a project may try to recover from the same blocked gap.

The runtime already refuses to repeat an action with identical declared inputs.
That is not the same guarantee. A gap like "no GPU and no private dataset" is a
property of the world, not of a sentence, and the old key hashed the sentence:
retiring the blocked question, giving the replacement a new id and rewording it
produced a fresh key, so the same impossible acquisition could be attempted for
as long as the budget lasted. Every attempt failed the same way and each one
looked new.

This module gives a blocked gap an identity the project cannot rename its way
out of, counts recorded attempts against it, and stops offering recovery once
the ceiling is reached. Two things — and only two — legitimately reset that
count, because both are new feasibility information rather than new wording:

* a capability the kernel has actually admitted since the attempts, and
* an attested world that did not exist when they were made.

A different question id, a reworded blocker, a new idea or the same error seen
again change nothing. Ruling a recovery path out is feasibility information, not
a scientific finding, and this module never records it as one.

When recovery is exhausted and nothing else can run, :func:`handoff_package`
writes down what a later session (or a person with the missing instrument) would
need: the questions still open, the evidence and failures already in hand, the
resources that are missing, the next testable piece of work, and the conditions
under which the project could resume. It is a local artifact. Nothing is sent
anywhere, and no new action type exists to send it.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

RULE_VERSION = "recovery-budget-v1"

#: Recovery attempts allowed per substantive gap under one feasibility version.
DEFAULT_ATTEMPT_CEILING = 2

#: Action types that exist to remove a blocker rather than to measure something.
RECOVERY_ACTIONS = ("ACQUIRE", "PROJECT_PIVOT")

#: Endings that say something about the service rather than about the research.
#: They never count towards either bound.
TRANSPORT_STATUSES = frozenset({"service_unavailable", "budget_exhausted"})

#: An attempt only counts against the ceiling when it came back without
#: removing the gap. A successful acquisition is not a recovery attempt spent.
FAILED_STATUSES = frozenset({"blocked", "failed", "refused", "error",
                             "capability_gap", "resource_blocked"})

_NOISE = re.compile(r"[^a-z0-9:_]+")
_REVISION = re.compile(r"\b(rev|revision|v|version|attempt|take)[ _-]*\d+\b")


def normalize_gap(text: Any) -> str:
    """Strip the parts of a gap description a rewrite can change."""
    value = str(text or "").casefold()
    value = _REVISION.sub(" ", value)
    return _NOISE.sub("_", value).strip("_")


def _question(state: Any, qid: str) -> Mapping[str, Any]:
    rows: Iterable[Mapping[str, Any]] = (
        list(getattr(state, "blocked_questions", []) or [])
        + list(getattr(state, "open_questions", []) or [])
        + list(getattr(state, "partial_questions", []) or []))
    return next((row for row in rows if str(row.get("id") or "") == str(qid or "")), {})


def gap_of(state: Any, action: Any) -> str:
    """The substantive shortfall this action would try to remove."""
    extra = dict(getattr(action, "extra", {}) or {})
    parts = [str(extra.get("gap") or ""), str(extra.get("capability_name") or "")]
    # The declared gap can be as coarse as "observable". What the frontier
    # recorded as the blocker is what distinguishes one shortfall from another,
    # so both go into the identity and neither alone defines it.
    for qid in (getattr(action, "frontier_target_id", ""), getattr(action, "target", "")):
        row = _question(state, str(qid or ""))
        if row:
            parts.append(str(row.get("blocked_reason") or ""))
            break
    parts = [item for item in parts if item]
    if not parts:
        parts = sorted(str(item) for item in getattr(state, "missing_capabilities", []) or [])
    return normalize_gap(":".join(parts))


def feasibility_version(state: Any) -> dict[str, Any]:
    """What the project can currently execute with, as an identity.

    Only admitted capability and attested worlds appear here. A model's opinion
    that the gap might now be closable is not feasibility information.
    """
    worlds = sorted({
        str(row.get("digest") or row.get("world_digest") or row.get("id") or "")
        for row in getattr(state, "world_versions", []) or []
        if row.get("attested") is not False})
    return {
        "capabilities": sorted(str(name) for name in
                               getattr(state, "available_capabilities", []) or []),
        "worlds": [item for item in worlds if item],
    }


def recovery_key(state: Any, action: Any) -> str:
    """A gap's identity: what is missing, and what the project can run today."""
    gap = gap_of(state, action)
    if not gap:
        return ""
    payload = {
        "rule_version": RULE_VERSION,
        "action": str(getattr(action, "action_type", "") or ""),
        "gap": gap,
        "feasibility": feasibility_version(state),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def is_recovery(state: Any, action: Any) -> bool:
    if str(getattr(action, "action_type", "") or "") not in RECOVERY_ACTIONS:
        return False
    return bool(gap_of(state, action))


def attempts(state: Any, key: str) -> list[dict[str, Any]]:
    if not key:
        return []
    return [dict(row) for row in getattr(state, "artifacts", []) or []
            if row.get("kind") == "action_attempt"
            and str(row.get("recovery_key") or "") == key
            and str(row.get("status") or "") in FAILED_STATUSES]


def exhausted(state: Any, action: Any, *, ceiling: int = DEFAULT_ATTEMPT_CEILING) -> bool:
    """Has this gap already used its recovery attempts under this feasibility?"""
    if not is_recovery(state, action):
        return False
    return len(attempts(state, recovery_key(state, action))) >= max(0, int(ceiling))


def recovery_status(state: Any, action: Any,
                    *, ceiling: int = DEFAULT_ATTEMPT_CEILING) -> dict[str, Any]:
    key = recovery_key(state, action)
    spent = len(attempts(state, key))
    return {
        "rule_version": RULE_VERSION,
        "recovery_key": key,
        "gap": gap_of(state, action),
        "attempts_spent": spent,
        "attempts_allowed": max(0, int(ceiling)),
        "exhausted": bool(key) and spent >= max(0, int(ceiling)),
        "feasibility": feasibility_version(state),
        "resets_on": "an admitted capability or an attested world that did not exist before",
        "note": ("ruling a recovery path out is feasibility information about "
                 "this project, not a scientific result"),
    }


# --------------------------------------------------------------------------
# Turns that keep producing the same outcome while nothing moves
# --------------------------------------------------------------------------

#: What a turn could have moved. Deliberately built from *kinds and reasons*,
#: never from the ids carrying them: three proposals refused for the same
#: registration error are the same situation, however many new idea ids the
#: project minted along the way.
def gap_signature(state: Any) -> str:
    frontier = dict(getattr(state, "frontier", {}) or {})
    quality = dict(getattr(state, "quality", {}) or {})
    answer = dict(quality.get("question_answer") or {})
    measurement = sorted({
        f"{row.get('kind')}:{normalize_gap(row.get('gap'))}"
        for row in frontier.get("measurement_gaps") or ()})
    readings = sum(
        1 for row in getattr(state, "evidence_records", []) or []
        if row.get("epistemic") == "WORLD" and not row.get("invalidated")
        and row.get("role") == "ProbeEvidence")
    payload = {
        "rule_version": RULE_VERSION,
        "measurement_gaps": measurement,
        "quality_gap": sorted(str(item) for item in quality.get("quality_gap") or ()),
        "answer": f"{answer.get('status', '')}/{answer.get('strength', '')}",
        "world_readings": readings,
        "capabilities": feasibility_version(state)["capabilities"],
        "blockers": sorted({normalize_gap(row.get("blocked_reason"))
                            for row in getattr(state, "blocked_questions", []) or ()}),
        # A recorded judgment and a recorded design revision are situation
        # changes in their own right: both gate readiness, and both come from
        # actions that cost nothing. Counting them keeps the bound off the
        # review loop it was never aimed at, while the paid generators — whose
        # output lands in other artifact kinds — stay covered.
        "recorded_reviews": len(list(getattr(state, "reviews", []) or ())),
        "design_revisions": len(list(getattr(state, "design_revisions", []) or ())),
        # A verification does not add a reading; it changes what an existing one
        # is allowed to support. Diagnosing a negative pilot is what unlocks the
        # confirmation after it, so it has to count as movement or the step that
        # follows could be refused for standing still.
        "verification": sorted(
            f"{row.get('evidence_id')}:{int(bool(row.get('diagnosed')))}"
            f"{int(bool(row.get('replay_ok')))}{int(bool(row.get('invalidated')))}"
            for row in getattr(state, "evidence_records", []) or ()
            if row.get("role") == "ProbeEvidence"),
        # A registered hypothesis and a project epoch are the other two changes
        # a free or near-free turn can make that the gap list does not show.
        # Discovery *ideas* are deliberately absent: minting one is exactly the
        # move this bound exists to stop counting as progress.
        "registered_theories": len(list(getattr(state, "theories", []) or ())),
        "project_epoch": int(getattr(state, "project_epoch", 0) or 0),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def stall_key(state: Any, action: Any) -> str:
    """This action, on this question, facing this unchanged situation."""
    payload = {
        "rule_version": RULE_VERSION,
        "action": str(getattr(action, "action_type", "") or ""),
        "question": str(getattr(action, "frontier_target_id", "") or ""),
        "gaps": gap_signature(state),
    }
    if payload['action'] == 'EXPLORE':
        # Two different programs returning "observed" are not two identical
        # answers. Exploratory measurements intentionally do not promote WORLD
        # evidence, so the scientific gap signature alone never distinguishes
        # them. Bind the same concrete inputs used by the execution deduplicator;
        # changing a title or idea ID does not reset this bound.
        from .exploratory import environment
        from ...services.experiments import fingerprint
        idea = next((r for r in getattr(state, 'artifacts', [])
                     if r.get('kind') == 'discovery_idea' and r.get('id') == action.target), {})
        spec = (idea.get('idea') or {}).get('exploration') or {}
        payload['exploration_inputs'] = fingerprint({
            'code': spec.get('code'), 'args': spec.get('args', {}),
            'assets_digest': environment(state).get('assets_digest')})
    elif payload['action'] == 'IDEATE':
        # New executed observations deserve a response even if they have not
        # closed a scientific gate. Otherwise two exploratory decisions stall
        # every later decision. Failed attempts alone must not reset the bound.
        from .response import result_packet
        packet = result_packet(state, str(getattr(action, 'target', '') or ''))
        observations = [{'result_id': r['result_id'], 'digest': r['digest']}
                        for r in packet['results'] if r['kind'] in {
                            'unvalidated_exploration', 'independent_native_evaluation',
                            'experiment_record', 'literature_result', 'dependency_retraction'}]
        if observations:
            payload['decision_observations'] = observations
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def stalled(state: Any, action: Any, *, ceiling: int = DEFAULT_ATTEMPT_CEILING) -> bool:
    """Has this turn already been taken, to the same effect, with nothing moving?

    The runtime refuses to repeat an action with identical declared inputs. That
    check reads the inputs, so a generator could propose a different-looking idea
    that failed for the identical recorded reason, and do it again, and again —
    which is what a measured live run spent most of its budget on. This reads the
    *situation* instead: same action, same question, same recorded gaps, same
    outcome. Two of those is enough; a third has nothing new to find out.

    A turn is never blocked by its own first attempt, and the moment any recorded
    gap moves — a reading admitted, a gap closed, a capability arriving — the
    signature changes and the action is available again.
    """
    key = stall_key(state, action)
    # An outage is not an outcome. A turn that came back `service_unavailable` or
    # `budget_exhausted` learned nothing about the science, and a later session
    # with the service restored must be free to try it again — the same exemption
    # `unattempted_actions` already makes for its own key.
    same = [row for row in getattr(state, "artifacts", []) or []
            if row.get("kind") == "action_attempt"
            and str(row.get("stall_key") or "") == key
            and str(row.get("status") or "") not in TRANSPORT_STATUSES]
    if len(same) < max(1, int(ceiling)):
        return False
    outcomes = {str(row.get("outcome_key") or "") for row in same[-int(ceiling):]}
    return len(outcomes) == 1


def _texts(rows: Iterable[Mapping[str, Any]], key: str = "text") -> list[str]:
    return [str(row.get(key) or row.get("id") or "") for row in rows]


def _judged_short(state: Any) -> dict[str, Any]:
    from .judgment import judged_short

    return judged_short(state)


def handoff_package(state: Any, *, workspace: Any = None,
                    reason: str = "") -> dict[str, Any]:
    """What a later session or a better-equipped lab would need to continue.

    Nothing here is a plan the project could have run and did not. When there is
    no executable experiment left, this says which instrument is missing rather
    than shipping a script that would pretend to run without it.
    """
    quality = dict(getattr(state, "quality", {}) or {}) or {}
    answer = dict(quality.get("question_answer") or {})
    evidence = [row for row in getattr(state, "evidence_records", []) or []
                if row.get("epistemic") == "WORLD" and not row.get("invalidated")]
    failures = [{"action_type": str(row.get("action_type") or ""),
                 "reason": str(row.get("reason") or row.get("status") or "")}
                for row in getattr(state, "artifacts", []) or []
                if row.get("kind") == "action_attempt"
                and str(row.get("status") or "") in FAILED_STATUSES]
    blocked = list(getattr(state, "blocked_questions", []) or [])
    missing = sorted(str(item) for item in getattr(state, "missing_capabilities", []) or [])
    spent_gaps = sorted({str(row.get("recovery_key") or "") for row in
                         getattr(state, "artifacts", []) or []
                         if row.get("kind") == "action_attempt" and row.get("recovery_key")})
    resources = dict(getattr(state, "resources", {}) or {})
    return {
        "kind": "research_handoff",
        "rule_version": RULE_VERSION,
        "reason": str(reason or "recovery_exhausted_and_no_executable_action"),
        "workspace": str(workspace or ""),
        "mission": str(getattr(state, "goal", "") or ""),
        "unanswered_questions": _texts(list(getattr(state, "open_questions", []) or []) + blocked),
        "blocked_questions": [{"id": str(row.get("id") or ""), "text": str(row.get("text") or ""),
                               "blocked_reason": str(row.get("blocked_reason") or "")}
                              for row in blocked],
        "answer_so_far": {"status": answer.get("status", "unresolved"),
                          "strength": answer.get("strength", "missing"),
                          "evidence_ids": list(answer.get("evidence_ids") or []),
                          "unresolved_reason": str(answer.get("unresolved_reason") or ""),
                          "unlock": str(answer.get("unlock") or "")},
        # Confirmed in its own scope and not promoted: a later session should
        # not have to re-measure what this one already established.
        "operational_findings": [dict(row) for row in answer.get("operational_findings") or ()],
        "candidate_mission_claim": dict(answer.get("candidate_mission_claim") or {}),
        # Novelty and importance as a panel judged them, where it judged them
        # short: the grade, the argument that decided it and what the
        # adjudicator said would raise it. A judgment, never a measurement.
        "judgment": _judged_short(state),
        "evidence_in_hand": [{"evidence_id": str(row.get("evidence_id") or ""),
                              "outcome": str(row.get("outcome") or ""),
                              "study_stage": str(row.get("study_stage") or ""),
                              "world_id": str(row.get("world_id") or "")}
                             for row in evidence],
        "recorded_failures": failures,
        "missing_resources_or_instruments": missing,
        # Registrations that cannot run and studies that have not covered every
        # setting. Where a run stopped before it ever named a capability, this is
        # what the next session would otherwise have to rediscover.
        "measurement_gaps": [dict(row) for row in
                             (getattr(state, "frontier", {}) or {}).get("measurement_gaps") or ()],
        "exhausted_recovery_gaps": spent_gaps,
        "next_testable_work": [
            f"{name}: {', '.join(sorted(set(dim.get('required_next_work') or [])))}"
            for name, dim in sorted((quality.get("dimensions") or {}).items())
            if dim.get("required_next_work") and name in set(quality.get("quality_gap") or [])
        ],
        "budget_requirement": {
            "remaining": dict(resources.get("remaining") or {}),
            "spent": dict(resources.get("spent") or {}),
            "exhausted": list(resources.get("exhausted") or []),
        },
        "resume_conditions": [
            "an admitted capability or attested world that covers a listed missing instrument",
            "a budget grant recorded in the contract, which resuming never replenishes by itself",
        ],
        "scope": "local_artifact_only_nothing_was_sent_to_anyone",
    }
