"""Decision journal and outcome contingencies: the compact project brief.

A worker used to receive ``deepcopy(state.to_dict())`` — the entire scientific
state, growing without bound — and a ``projection`` argument that no worker ever
read. Over a long trajectory that is the worst of both worlds: the prompt is
dominated by accumulated narrative, and the carefully-computed relevant slice is
thrown away.

The journal is a *derived* view. It is recomputed from canonical state every
time it is asked for, it is versioned, and nothing reads it as authority. It
answers the five questions a researcher would actually want answered before
choosing the next action:

    what is this project for
    what is the current strategy and why is it still viable
    what is the strongest thing standing against it
    what is the next decisive action
    what is left to spend

Contingency planning is the other half. A long-horizon agent should know before
running an experiment what each outcome would mean — and should say so in
branches, not in a fabricated ``P(success) = 0.82``. Nothing in this runtime is
a calibrated predictor, so nothing here emits a probability.
"""

from __future__ import annotations

import json

import copy
from typing import Any, Mapping

from .contract import contract_from_state
from .quality import ReviewRecord, open_reviews
from .resources import budget_phase, budget_state


JOURNAL_VERSION = "decision-journal-v2"

MAX_EVIDENCE = 6
MAX_FAILURES = 4
MAX_OBJECTIONS = 5


def _relevant_evidence(state: Any, *, question_id: str = "", theory_id: str = "") -> list[dict[str, Any]]:
    rows = [
        row for row in getattr(state, "evidence_records", []) or []
        if str(row.get("epistemic") or "") == "WORLD" and row.get("attested")
        and not row.get("invalidated") and not row.get("promotion_refused")
    ]
    if theory_id:
        focused = [row for row in rows if str(row.get("theory_id") or "") == theory_id]
        rows = focused or rows
    if question_id:
        focused = [row for row in rows if str(row.get("question_id") or "") == question_id]
        rows = focused or rows
    return [
        {
            key: row.get(key)
            for key in ("evidence_id", "theory_id", "question_id", "outcome", "verdict",
                        "treatment", "control", "sample_size", "world_id",
                        "reproduction_ok", "transfer_ok", "cannot_corroborate",
                        "promotion_ok", "study_stage", "pilot_verdict", "stage_verdict",
                        "world_digest", "registration_digest", "execution_status",
                        "primary_metric", "sample_size_basis", "inference_link",
                        "validity", "failure_diagnosis")
            if row.get(key) is not None
        }
        for row in rows[-MAX_EVIDENCE:]
    ]


def current_strategy(state: Any) -> dict[str, Any]:
    """What the project is currently trying, and whether it still holds up."""
    contract = contract_from_state(state)
    quality = getattr(state, "quality", {}) or {}
    dimensions = quality.get("dimensions") or {}
    live = [
        row for row in getattr(state, "theories", []) or []
        if row.get("status") not in {"retired"} and row.get("role") != "competing_explanation"
    ]
    gap = list(quality.get("quality_gap") or [])
    fatal = open_reviews(getattr(state, "reviews", []) or [], severity="fatal")
    revisions = [
        row for row in getattr(state, "design_revisions", []) or []
        if int(row.get("project_epoch") or 0) == int(getattr(state, "project_epoch", 0) or 0)
    ]
    viable: list[str] = []
    doubts: list[str] = []
    if live:
        viable.append(f"{len(live)} live hypotheses still attached to the contract")
    else:
        doubts.append("no live hypothesis is attached to the root question")
    if fatal:
        doubts.append(f"{len(fatal)} fatal objections stand unrepaired")
    if len(revisions) >= 2 and all(str(r.get("outcome") or "open") != "resolved" for r in revisions):
        doubts.append(f"{len(revisions)} redesigns have not moved their dimension")
    if int((getattr(state, "debt", {}) or {}).get("goal_drift_debt") or 0):
        doubts.append("open questions have drifted off the root question")
    if not gap:
        viable.append("the contract's quality requirements are met")
    return {
        "root_question": contract.root_question,
        "target_quality": contract.target_quality,
        "project_epoch": int(getattr(state, "project_epoch", 0) or 0),
        "live_hypotheses": [str(row.get("id") or "") for row in live][-6:],
        "quality_gap": gap,
        "weakest_dimensions": sorted(
            gap, key=lambda name: str((dimensions.get(name) or {}).get("status") or ""))[:3],
        "why_still_viable": viable,
        "why_it_might_not_be": doubts,
        "verdict": "viable" if viable and not doubts else (
            "under_doubt" if viable else "not_viable_as_stated"),
    }


def next_decisive_action(state: Any) -> dict[str, Any]:
    """Legacy name for one unranked dependency-based planning suggestion.

    This lookup does not estimate informativeness or select the scientific
    direction. The model must interpret evidence and remaining alternatives.
    """
    quality = getattr(state, "quality", {}) or {}
    dimensions = quality.get("dimensions") or {}
    fatal = open_reviews(getattr(state, "reviews", []) or [], severity="fatal")
    major = open_reviews(getattr(state, "reviews", []) or [], severity="major")
    if fatal:
        record = ReviewRecord.from_dict(fatal[0])
        return {
            "action": "REDESIGN", "because": record.objection,
            "review_id": record.review_id, "repair_type": record.repair_type,
            "decisive_because": "recorded objection requests a revised design; interpretation remains open",
        }
    for name in list(quality.get("quality_gap") or []):
        work = (dimensions.get(name) or {}).get("required_next_work") or []
        if work:
            return {
                "action": str(work[0]), "because": f"{name} is below the contract",
                "dimension": name,
                "blocking_reasons": list((dimensions.get(name) or {}).get("blocking_reasons") or []),
                "decisive_because": f"{name} has an outstanding recorded requirement",
            }
    if major:
        record = ReviewRecord.from_dict(major[0])
        return {"action": "REDESIGN", "because": record.objection,
                "review_id": record.review_id, "repair_type": record.repair_type,
                "decisive_because": "an unrepaired major objection caps the claim"}
    return {"action": "", "because": "no outstanding structural dependency recorded",
            "decisive_because": "mission coverage and scientific interpretation remain separate"}


def research_journal(
    state: Any, *, question_id: str = "", theory_id: str = ""
) -> dict[str, Any]:
    """The compact, derived project brief a worker should see instead of history."""
    contract = contract_from_state(state)
    reviews = getattr(state, "reviews", []) or []
    budget = budget_state(state)
    contradictions = list(getattr(state, "contradictions", []) or [])
    failures = list(getattr(state, "failed_designs", []) or [])[-MAX_FAILURES:]
    decisions = list(getattr(state, "decision_journal", []) or [])[-6:]
    questions = [copy.deepcopy(q) for key in ("open_questions", "resolved_questions",
                 "partial_questions", "blocked_questions") for q in getattr(state, key, []) or []]
    beliefs = copy.deepcopy(dict(getattr(state, "belief_states", {}) or {}))
    # This history is intentionally independent of recent prompt windows. A
    # retired hypothesis and its frozen refutation remain relevant after any
    # amount of later proposal/critic prose. An invalidated result stays visible
    # as invalidated, never as a still-valid refutation.
    history = []
    for row in getattr(state, "evidence_records", []) or []:
        if row.get("epistemic") != "WORLD":
            continue
        eid = str(row.get("evidence_id") or "")
        belief = beliefs.get(str(row.get("theory_id") or ""), {})
        if row.get("invalidated"):
            status = "invalidated"
        elif eid in belief.get("contradicting_evidence", []):
            status = "scoped_refutation"
        elif (eid in belief.get("observation_evidence", [])
              and row.get("outcome") in {"weakens", "negative", "contradicted"}):
            status = "limited_negative_observation"
        else:
            continue
        history.append({
            "evidence_id": eid, "theory_id": str(row.get("theory_id") or ""),
            "question_id": str(row.get("question_id") or ""), "status": status,
            "recorded_outcome": str(row.get("outcome") or row.get("verdict") or ""),
            "world_id": str(row.get("world_id") or ""),
            "world_digest": str(row.get("world_digest") or ""),
            "experiment_digest": str(row.get("experiment_digest") or ""),
            "invalidation_reason": str(row.get("invalidation_reason") or ""),
            "prohibited_inferences": list(row.get("prohibited_inferences") or []),
        })
    critical_gaps = [
        {"question_id": q.get("id"), "kind": "remaining_mission_gap", "reason": gap}
        for q in questions for gap in q.get("remaining_mission_gaps") or []]
    critical_gaps.extend({"theory_id": tid, **dict(gap)} for tid, belief in beliefs.items()
                         for gap in belief.get("unresolved_gaps") or [])
    critical_gaps.extend({"question_id": q.get("id"), "kind": "question_basis",
                          "reason": "reconsideration_required"} for q in questions
                         if q.get("question_basis_status") == "reconsideration_required"
                         or q.get("transition_status") == "reconsideration_required")
    # Why a registered experiment currently cannot run, in the brief the
    # generator and the critic both read. A measured live run proposed the same
    # refused study three times because this was recorded in state and shown to
    # nobody: the loop could see the refusal, and the only party who could
    # repair it could not.
    rejected = []
    for row in (getattr(state, "frontier", {}) or {}).get("measurement_gaps") or ():
        entry = {"idea_id": row.get("idea_id"), "question_id": row.get("question_id"),
                 "kind": row.get("kind"), "reason": row.get("gap"),
                 "details": list(row.get("errors") or [])}
        instrument = row.get("instrument")
        if isinstance(instrument, Mapping):
            # A live proposal described connectivity survival in words, wrote no
            # protocol, and was never told the instrument it described exists.
            available = dict(instrument.get("available") or {})
            entry["instruments_available_for_this_world_class"] = dict(
                available.get("operations") or {})
            entry["how_to_proceed"] = (
                "register an experiment with one of these observables if it measures what "
                "the claim needs; if none does, say so rather than proposing an unmeasurable claim"
                if available.get("world_class_supported", True) else
                "no instrument exists for this kind of world; choose a world class that has one "
                f"({', '.join(available.get('known_world_classes') or ())}) or state the gap")
        rejected.append(entry)
    critical_gaps.extend(dict(row, kind=str(row["kind"])) for row in rejected)
    suggestion = next_decisive_action(state)
    suggestion.update(is_ranked=False, basis="recorded_dependency_candidate",
                      requires_scientific_interpretation=True)
    from .design import research_design_feedback
    from .exploratory import environment
    from .dependencies import dependency_view
    from .termination import task_delivery
    return {
        "version": JOURNAL_VERSION,
        "exploration_environment": environment(state),
        "result_dependencies": dependency_view(state),
        "task_delivery": task_delivery(state),
        "tool_policy": next((dict(r) for r in reversed(state.artifacts)
                             if r.get('kind') == 'research_tool_policy'), {}),
        "native_evaluator": next((dict(r) for r in reversed(state.artifacts)
                                  if r.get('kind') == 'native_evaluation_environment'), {}),
        "experiment_design_feedback": research_design_feedback(state),
        "is_derived_projection": True,
        "not_authoritative": "recomputed from ScientificState; never a second truth",
        "research_contract": contract.to_dict(),
        "strategy": current_strategy(state),
        "next_decisive_action": suggestion,
        "current_questions": questions,
        "categorical_beliefs": beliefs,
        "remaining_critical_gaps": critical_gaps,
        # Listed separately as well, because "propose something else" is the
        # wrong answer to "this registration was refused for this reason".
        "registrations_that_cannot_run": rejected,
        "refutation_and_invalidation_history": history,
        "relevant_world_evidence": _relevant_evidence(
            state, question_id=question_id, theory_id=theory_id),
        "strongest_contradiction": contradictions[0] if contradictions else None,
        "open_objections": [
            {k: row.get(k) for k in
             ("review_id", "severity", "dimension", "objection", "repair_type",
              "requires_new_evidence")}
            for row in open_reviews(reviews)
            if ReviewRecord.from_dict(row).severity in {"fatal", "major"}
        ][:MAX_OBJECTIONS],
        "recent_failures": [
            {k: row.get(k) for k in ("action_type", "reason", "theory_id") if row.get(k)}
            for row in failures
        ],
        "recorded_decisions": decisions,
        "resources": {
            "phase": budget_phase(state),
            "remaining": dict(budget.get("remaining") or {}),
            "budget_pressure": budget.get("budget_pressure"),
        },
        "epistemic_reminders": [
            "reviewer judgment is not evidence",
            "a GENERATED hypothesis never becomes WORLD by being restated",
            "an objection requiring new evidence cannot be closed by rewriting",
        ],
    }


# ---------------------------------------------------------------------------
# The result a run hands back
# ---------------------------------------------------------------------------


def render_result(state: Any) -> str:
    """What was tested, what was read, and what that does and does not establish.

    A live run reached RESEARCH_READY on a confirmed measurement and handed its
    caller an IDEA.md headed "a research proposal, not evidence" with no number
    in it, beside a status page saying both "ideas that have not been validated
    yet" and "answer: supported". The proposal and the critic's view of it each
    had a document; the *result* had none, so the one thing the run existed to
    produce was the one thing it did not write down.

    Everything here is projected from admitted state — the registration, the
    readings, the derived answer, the recorded gaps. No model is asked anything
    and nothing here can promote a claim: a reading appears with the outcome the
    kernel derived for it, under the scope the kernel gave it.
    """
    from .termination import readiness, task_delivery

    verdict = readiness(state)
    quality = dict(getattr(state, "quality", {}) or {})
    answer = dict(quality.get("question_answer") or {})
    readings = [dict(row) for row in getattr(state, "evidence_records", []) or []
                if row.get("role") == "ProbeEvidence" and row.get("epistemic") == "WORLD"]
    lines = [
        f"# RESULT — {getattr(state, 'goal', '') or 'untitled mission'}\n",
        "> Projected from admitted evidence in ScientificState. No model wrote any "
        "of this, and nothing here extends a result past the conditions it was "
        "measured under.\n",
        "## Verdict\n",
        f"- Run: **{verdict.status}** (contract scope only; `publication_ready: false`)",
        f"- Answer to the mission: **{answer.get('status', 'unresolved')}**"
        f" · strength `{answer.get('strength', 'missing')}`",
    ]
    if answer.get("unresolved_reason"):
        lines.append(f"- Why unresolved: `{answer['unresolved_reason']}`")
    if answer.get("unlock"):
        lines.append(f"- What would unlock it: {answer['unlock']}")
    lines.append(f"- Scope: `{answer.get('scope', 'registered_claim_under_tested_conditions_only')}`\n")
    delivery = task_delivery(state)
    if delivery['status'] != 'not_configured':
        lines.append(f"- Task delivery: **{delivery['status']}**; scientific confirmation "
                     f"under its separate contract: **{verdict.ready}**. "
                     "Submission acceptance does not establish a mechanism.\n")

    from .exploratory import RESULT_KIND
    explorations = [r for r in getattr(state, 'artifacts', []) if r.get('kind') == RESULT_KIND]
    if explorations:
        lines.append("## Exploratory program observations — scientific support not assessed\n")
        for r in explorations:
            run = r.get('run') or {}
            lines.append(f"- `{r['id']}`: {r.get('status')}; question: {(r.get('proposal') or {}).get('question', '')}")
            lines.append(f"  Printed measurements: `{json.dumps(run.get('result', {}), sort_keys=True)}`; "
                         f"failure: `{run.get('failure', '')}`. Scientific and mechanism verdicts: not assessed.")
            audit = r.get('measurement_audit') or {}
            if audit:
                lines.append(f"  Independent measurement checks: `{audit.get('status')}` "
                             "(declared quantities only; header and unit semantics remain unverified).")
                for check in audit.get('checks', []):
                    lines.append(f"  - `{check['check_id']}` / `{check.get('metric', '')}`: "
                                 f"{check['status']}; expected `{check.get('expected')}`, "
                                 f"observed `{check.get('observed')}`.")
    native = [r for r in getattr(state, 'artifacts', []) if r.get('kind') == 'native_evaluation']
    if native:
        from .dependencies import dependency_view
        dependency_status = dependency_view(state)
        lines.append('## Independent native evaluations\n')
        for r in native:
            lines.append(f"- `{r['id']}`: {r['status']}; dependency status "
                         f"`{dependency_status.get(r['id'], {}).get('status')}`; "
                         f"metrics `{json.dumps(r.get('metrics', {}), sort_keys=True)}`. "
                         f"Scope: {r.get('scope', 'unavailable')}. Mechanism support: not assessed.")
            for name, metric in r.get('metric_feedback', {}).items():
                lines.append(f"  - `{name}` = {metric['value']}: {metric.get('meaning')}. "
                             f"Scope: {metric.get('population')}; {metric.get('limitations', '')}")
        for r in getattr(state, 'artifacts', []):
            if r.get('kind') == 'native_comparison' and r.get('manipulation_check'):
                lines.append(f"- Intervention `{r['id']}`: `{json.dumps(r['manipulation_check'], sort_keys=True)}`")
    lines.append("## What was tested\n")
    registrations: dict[str, dict[str, Any]] = {}
    for row in readings:
        registration = row.get("registration") if isinstance(row.get("registration"), Mapping) else {}
        digest = str(row.get("registration_digest") or "")
        if digest and digest not in registrations:
            registrations[digest] = dict(registration)
    if not registrations:
        lines.append("No registered experiment produced an admitted reading.\n")
    for digest, registration in registrations.items():
        execution = dict(registration.get("execution") or {})
        lines.extend([
            f"- Registered claim: {registration.get('hypothesis') or '—'}",
            f"  - operation `{execution.get('operation')}` · metric `{execution.get('metric')}` · "
            f"direction `{execution.get('expected_direction')}` · margin `{execution.get('margin')}`",
            f"  - parameters `{json.dumps(execution.get('parameters'), sort_keys=True)}`",
            f"  - registration `{digest[:16]}…` · protocol `{registration.get('version')}`",
        ])
        study = execution.get("study") if isinstance(execution.get("study"), Mapping) else None
        if study:
            lines.append(f"  - study: {len(study.get('settings') or [])} registered settings · "
                         f"aggregation `{(study.get('aggregation') or {}).get('method')}` · "
                         f"scope: {study.get('coverage_scope')}")
    lines.append("")

    lines.append("## Readings\n")
    if readings:
        lines.extend(["| stage | setting | treatment | control | outcome | can corroborate | evidence |",
                      "|---|---|---:|---:|---|---|---|"])
        for row in readings:
            validity = row.get("validity") if isinstance(row.get("validity"), Mapping) else {}
            lines.append(
                f"| {row.get('study_stage') or '—'} | {row.get('study_setting_id') or '—'} | "
                f"{row.get('treatment')} | {row.get('control')} | {row.get('outcome')} | "
                f"{'yes' if validity.get('corroborates') else 'no'}"
                f"{' (invalidated)' if row.get('invalidated') else ''} | "
                f"`{str(row.get('evidence_id') or '')[:20]}` |")
    elif explorations or native:
        lines.append("No admitted WORLD readings. Exploratory observations and native scores, "
                     "when listed above, retain their own measurement scope.")
    else:
        lines.append("None. Nothing was measured, and nothing below should be read as a finding.")
    lines.append("")

    lines.append("## What this establishes, and what it does not\n")
    relations = dict(answer.get("relations") or {})
    for tid, relation in sorted(relations.items()):
        lines.append(f"- `{tid}` stands to the mission as **{relation.get('relation')}** "
                     f"({relation.get('basis')}, rule `{relation.get('rule_version')}`)")
    for finding in answer.get("operational_findings") or ():
        standing = ("the answer above rests on it" if finding.get("promoted_to_mission")
                    else "**not promoted to the mission**")
        settled = "Refuted" if finding.get("outcome") == "weakens" else "Confirmed"
        lines.append(f"- {settled} in its own scope ({standing}): "
                     f"{finding.get('claim')} → `{finding.get('outcome')}`")
    candidate = dict(answer.get("candidate_mission_claim") or {})
    if candidate:
        lines.append("- The mission has no structured definition. The comparison this run "
                     "measured is offered below for the caller to approve as "
                     "`contract.mission_claim`; approving it is the caller's decision, "
                     "and nothing in the runtime makes it:")
        lines.append(f"  `{json.dumps(candidate.get('mission_claim'), sort_keys=True)}`")
    studies = dict(answer.get("studies") or {})
    for tid, view in sorted(studies.items()):
        cover = dict(view.get("coverage") or {})
        lines.append(f"- Study `{tid}`: {view.get('status')} · verdict `{view.get('verdict') or '—'}` · "
                     f"coverage {cover.get('coverage_count')}/{len(cover.get('settings_registered') or [])} "
                     f"settings on {cover.get('independent_unit_count')} independent unit(s). "
                     "Settings are coverage, not sample size.")
    prohibited = sorted({(str(item)[:157] + "…") if len(str(item)) > 160 else str(item)
                         for row in readings for item in row.get("prohibited_inferences") or ()})
    if prohibited:
        lines.append(f"- Not established by any reading here: {', '.join(prohibited)}")
    if not (relations or answer.get("operational_findings") or candidate or studies):
        lines.append("- No admitted WORLD reading establishes the mission. Native scores "
                     "describe the evaluated submissions; they do not establish a causal mechanism.")
    lines.append("")

    lines.extend(_judgment_lines(state, quality))

    lines.append("## What remains\n")
    gaps = list(quality.get("quality_gap") or [])
    lines.append(f"- Quality below contract: {', '.join(gaps) if gaps else 'none'}")
    for row in (getattr(state, "frontier", {}) or {}).get("measurement_gaps") or ():
        instrument = row.get("instrument") if isinstance(row.get("instrument"), Mapping) else None
        if instrument:
            available = instrument.get("available") or {}
            lines.append(f"- Instrument gap (`{row.get('idea_id')}`): asked to measure "
                         f"`{(instrument.get('requested') or {}).get('primary_metric')}`; "
                         f"available: {', '.join(available.get('observables') or ()) or 'no instrument for this world class'}")
        elif row.get("kind") in {"registration_gap", "coverage_gap"}:
            lines.append(f"- {row.get('kind')} (`{row.get('idea_id')}`): {row.get('gap')}")
    failed = [name for name, ok, _ in verdict.gates if not ok]
    lines.append(f"- Unmet readiness gates: {', '.join(failed) if failed else 'none'}")
    return "\n".join(lines) + "\n"


def _judgment_lines(state: Any, quality: Mapping[str, Any]) -> list[str]:
    """Novelty and importance as judged — by whom, how far apart, and on what.

    A grade is never printed without the panel that gave it: which roles sat,
    whether they were one model or several, what they could not agree on. A
    reader who sees "importance: partial" alone will take it for a finding.
    """
    from .judgment import SOURCE, judgment_subject, required_judgments

    lines = ["## How it was judged\n"]
    digest = str(judgment_subject(state).get("digest") or "")
    rows = {str(row.get("dimension") or ""): row for row in getattr(state, "reviews", []) or []
            if row.get("source") == SOURCE
            and str((row.get("panel") or {}).get("subject_digest") or "") == digest}
    required = required_judgments(state)
    if not rows:
        asked = (f"the contract requires {', '.join(sorted(required))}" if required
                 else "the contract does not ask for it")
        lines.append(f"Novelty and importance: nobody has judged this result ({asked}). "
                     "No measurement can settle either.\n")
        return lines
    dims = quality.get("dimensions") or {}
    lines += ["| dimension | graded | required | examiner | advocate | skeptic | adjudicator |",
              "|---|---|---|---|---|---|---|"]
    for name in sorted(rows):
        grades = (rows[name].get("panel") or {}).get("grades") or {}
        lines.append(f"| {name} | {(dims.get(name) or {}).get('status', '—')} | "
                     f"{required.get(name, '—')} | " + " | ".join(
                         str(grades.get(role) or "—")
                         for role in ("examiner", "advocate", "skeptic", "adjudicator")) + " |")
    panel = dict(next(iter(rows.values())).get("panel") or {})
    models = sorted(set((panel.get("models") or {}).values()))
    lines.append(f"\n- Panel: `{panel.get('version')}` · independence `{panel.get('independence')}` "
                 f"({', '.join(models) or 'model unrecorded'}). Roles served by one model agree "
                 "more than independent judges would.")
    for name in sorted(rows):
        row, detail = rows[name], dict(rows[name].get("panel") or {})
        if row.get("assessment_basis"):
            lines.append(f"- {name} — decisive argument: {row['assessment_basis']}")
        if detail.get("caps"):
            lines.append(f"- {name} — capped by: {', '.join(detail['caps'])}")
        if detail.get("improve"):
            lines.append(f"- {name} — what would raise it: {detail['improve']}")
    if panel.get("unresolved_disagreement"):
        lines.append(f"- Left unresolved: {panel['unresolved_disagreement']}")
    if panel.get("transcript"):
        lines.append(f"- Transcript: `{panel['transcript']}`")
    lines.append("- This is a judgment, not a measurement. It added no evidence and moved no answer.\n")
    return lines


# ---------------------------------------------------------------------------
# Outcome contingencies
# ---------------------------------------------------------------------------


def contingency_plan(state: Any, action: Any = None) -> dict[str, Any]:
    """What each outcome of the next experiment would mean, in branches.

    Deliberately not a forecast. There is no calibrated predictor in this
    runtime, so no branch carries a probability; each branch says what the
    project would do, and why, if that outcome occurred.
    """
    quality = getattr(state, "quality", {}) or {}
    dimensions = quality.get("dimensions") or {}
    adequacy = str((dimensions.get("experimental_adequacy") or {}).get("status") or "unknown")
    target = str(getattr(action, "target", "") or "")
    belief = dict((getattr(state, "belief_states", {}) or {}).get(target) or {})
    return {
        "kind": "outcome_contingency_plan",
        "target": target,
        "is_forecast": False,
        "no_probabilities_because": "this runtime has no calibrated outcome predictor",
        "power_state": "unknown",
        "registered_sample_floor_state": adequacy,
        "sample_floor_is_not_power": True,
        "scoped_belief_state": str(belief.get("status") or "insufficient"),
        "unresolved_gaps": copy.deepcopy(list(belief.get("unresolved_gaps") or [])),
        "branches": [
            {
                "outcome": "supports",
                "means": "the registered comparison points in the predicted direction",
                "does_not_mean": "the mechanism is established or the effect generalizes",
                "next": ["REVIEW: interpret the registered result with its measurement, sample and identification gaps",
                         "VERIFY: check the result within its recorded scope"],
            },
            {
                "outcome": "weakens",
                "means": "the registered comparison opposes the prediction; sample and identification checks still govern corroboration",
                "does_not_mean": "the hypothesis is refuted in general",
                "next": ["VERIFY: check execution and measurement before interpreting a negative observation",
                         "REVIEW: compare surviving explanations and unresolved assumptions; a negative sign alone selects no repair"],
            },
            {
                "outcome": "inconclusive",
                "means": "the arms did not separate beyond the registered margin",
                "does_not_mean": "a negative result",
                "next": [
                    "REVIEW: identify unresolved measurement, sample or discrimination gaps before choosing another experiment",
                ],
            },
        ],
    }
