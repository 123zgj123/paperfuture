"""Research frontier, debt, and a thin relation graph over ScientificState.

These are projections of the canonical state, not a second truth.
"""

from __future__ import annotations

from typing import Any, Mapping

from .state import ScientificState


SCIENTIFIC_ACTIONS = frozenset(
    {
        "SURVEY",
        "ASK",
        "OBSERVE",
        "ACQUIRE",
        "THEORIZE",
        "PROBE",
        "EXPLORE",
        "VERIFY",
    }
)

MAINTENANCE_ACTIONS = frozenset(
    {
        "REFLECT",
        "ARCHIVE",
        "EVOLVE_HARNESS",
        "SYNTHESIZE",
        "FORK_LINEAGE",
        # Project-level actions act on the project, not on one frontier node, so
        # they are not gated by the frontier-reachability check.
        "REVIEW",
        "REDESIGN",
        "PROJECT_PIVOT",
    }
)

PROJECT_ACTIONS = frozenset({"REVIEW", "REDESIGN", "PROJECT_PIVOT"})


def action_research_gap(state: ScientificState, action: Any) -> dict[str, Any] | None:
    """Bind an action to an unresolved dependency, without estimating its value.

    A dependency order is not information gain: competing feasible experiments
    still require scientific judgment. No worker-supplied priority is read.
    """
    kind, target, extra = action.action_type, action.target, action.extra
    questions = state.open_questions + state.partial_questions + state.blocked_questions + state.resolved_questions
    question = next((q for q in questions if q.get('id') == action.frontier_target_id), {})
    theory = next((t for t in state.theories if t.get('id') == target), {})
    eid = next((e for e in state.evidence_records if e.get('evidence_id') == target), {})
    idea_id = extra.get('discovery_idea_id') or extra.get('parent_idea_id')
    theories = [t for t in state.theories if t.get('discovery_idea_id') == idea_id] if idea_id else []
    evidence = [e for e in state.evidence_records if not e.get('invalidated') and
                (e.get('discovery_idea_id') == idea_id if idea_id else e.get('theory_id') == target)]
    observations = [e for e in evidence if e.get('epistemic') == 'WORLD' and e.get('attested')]
    phase, gap, basis = '', '', []
    if question.get('transition_status') == 'reconsideration_required' and kind not in {'REVIEW', 'IDEATE', 'REDESIGN'}:
        return None
    if kind == 'EXPLORE':
        from .exploratory import environment, records
        if environment(state) and not records(state, target):
            phase, gap = 'evidence_acquisition', 'agent_designed_exploration_unexecuted'
    elif kind == 'VERIFY':
        if not eid or eid.get('invalidated'):
            return None
        if not eid.get('verify_count') or extra.get('diagnostic') and not eid.get('diagnosed'):
            phase, gap, basis = 'verification', 'measurement_or_execution_unverified', [target]
        elif extra.get('verify_kind') in {'VERIFY_REPLICATION', 'VERIFY_TRANSFER'}:
            phase, gap, basis = 'replication', 'independent_validation_missing', [target]
    elif kind == 'PROBE':
        if any(t.get('status') in {'refuted', 'retired'} for t in theories) or theory.get('status') in {'refuted', 'retired'}:
            return None
        stage = str(extra.get('study_stage') or '')
        setting = str(extra.get('study_setting_id') or '')
        if stage == 'replication':
            phase, gap = 'replication', 'independent_validation_missing'
        elif stage == 'confirmatory':
            phase, gap = 'verification', 'scoped_prediction_unconfirmed'
        else:
            # Re-running an inconclusive freeze cannot improve power or
            # discrimination — but a reading only answers the stage and the
            # registered setting that produced it. Treating every reading as
            # "observed" let a calibration consume the pilot a scientific run
            # still needed, and let one setting's reading consume all the others:
            # the only action that could close the gap left the menu, and the
            # loop was left with nothing to do but propose another idea.
            answered = [e for e in observations
                        if str(e.get('study_stage') or '') == stage
                        and str(e.get('study_setting_id') or '') == setting]
            if not answered:
                phase, gap = 'evidence_acquisition', 'registered_prediction_untested'
        basis = [str(e.get('evidence_id')) for e in observations]
    elif kind == 'ACQUIRE':
        if extra.get('gap'):
            phase, gap = 'evidence_acquisition', str(extra['gap'])
            # A generic request to choose a world has no executable acquisition
            # under an explicit-only contract. Compile a question-specific
            # requirement first; do not spend a step on a known refusal.
            if (not extra.get('experiment_requirement') and not extra.get('capability_name')
                    and (state.contract or {}).get('world_selection', 'declared_only') == 'declared_only'
                    and gap == 'artifact_inventory'):
                return None
    elif kind == 'REVIEW':
        from .quality import result_review_scope
        scope = result_review_scope(state)
        if extra.get('judgment_panel'):
            # Behind every measurement: a panel judges what was found, and a
            # result still being measured is not yet a subject.
            phase, gap, basis = 'synthesis', 'required_judgment_unassessed', list(scope['evidence_ids'])
        elif extra.get('program_review'):
            phase, gap = 'discrimination', 'interpretation_or_feasibility_unresolved'
            basis = list((extra.get('review_answers') or {}).get('evidence_ids') or [])
        elif scope['evidence_ids'] and not any(r.get('result_review_scope') == scope['digest'] for r in state.reviews):
            phase, gap, basis = 'verification', 'current_result_scope_unreviewed', list(scope['evidence_ids'])
        elif extra.get('unreviewed_evidence') or extra.get('unreviewed_designs'):
            phase, gap = 'discrimination', 'experimental_design_unreviewed'
    elif kind == 'IDEATE':
        from .discovery import discovery_records
        ideas, _ = discovery_records(state)
        if not ideas:
            phase, gap = 'exploration', 'question_has_no_testable_hypothesis'
        elif extra.get('decision_request'):
            phase, gap = 'discrimination', str(extra['decision_request'])
            basis = [str(e.get('evidence_id')) for e in observations]
        elif not observations and not any(t.get('status') in {'refuted', 'retired'} for t in theories):
            phase, gap = 'exploration', str(extra.get('failure_class') or 'hypothesis_not_operationalized')
        elif extra.get('judged_below_the_bar'):
            # A verdict on the direction, so it is acted on before more is spent
            # polishing the result it was a verdict on.
            phase, gap = 'discrimination', 'judged_result_falls_short_of_the_contract'
            basis = [str(e.get('evidence_id')) for e in observations]
        elif extra.get('operator') in {'MEASUREMENT_REPAIR', 'DISCRIMINATIVE_TEST'} and not any(
                t.get('status') == 'refuted' for t in theories):
            phase, gap = 'discrimination', 'observation_does_not_distinguish_explanations'
            basis = [str(e.get('evidence_id')) for e in observations]
        elif extra.get('operator') and extra.get('parent_idea_id'):
            # Every other evolution the planner offers. Without this branch, once
            # anything had been measured only the two operators above could be
            # scheduled: across 712 live ticks IDEATE was dropped 44 times as
            # `no_unresolved_research_gap_addressed`, 36 of them the SCOPE_SHIFT
            # that exists for a settled result which leaves the question open,
            # and TRANSFER_STRUCTURE never ran in 117 live proposals. Last in
            # the order, because widening follows every named dependency — but
            # on the order, because a move that can never be chosen is not a move.
            phase, gap = 'exploration', str(extra.get('failure_class') or action.reason
                                            or 'idea_needs_revision')
            basis = [str(e.get('evidence_id')) for e in observations]
    elif kind == 'CRITIQUE':
        idea = next((a for a in state.artifacts if a.get('kind') == 'discovery_idea' and a.get('id') == target), None)
        if idea and not any(a.get('kind') == 'idea_feedback' and a.get('idea_id') == target for a in state.artifacts):
            phase, gap = 'exploration', 'proposal_design_unassessed'
            # A proposal that has already been *calibrated* is waiting on this
            # verdict: the scientific pilot is gated on it. Filed under
            # exploration it sat behind every free REVIEW, REDESIGN and
            # SYNTHESIZE — a live run calibrated a runnable experiment at tick 7
            # and did not critique it until tick 26, so the pilot never ran.
            #
            # The condition is the calibration reading, not mere runnability. A
            # first version ordered the critique of *every* runnable proposal
            # this way, and a nine-mission campaign priced it: three extra calls
            # on each mission that would have gone straight to its pilot, and one
            # mission sent into a loop of identical clones because the critic
            # said REVISE before anything had been measured. A proposal that can
            # go to its pilot without a verdict is not waiting on one.
            calibrated = any(
                e.get('discovery_idea_id') == target and e.get('study_stage') == 'calibration'
                and e.get('epistemic') == 'WORLD' and not e.get('invalidated')
                for e in state.evidence_records)
            if calibrated:
                phase, gap = 'evidence_acquisition', 'calibrated_proposal_awaits_assessment'
    elif kind == 'SURVEY':
        if extra.get('knowledge_request'):
            from .inquiry import inquiry_is_due
            if not inquiry_is_due(state, extra['knowledge_request']):
                return None
            return {'id': f'{target}:named_research_uncertainty',
                    'phase': 'context', 'kind': 'named_research_uncertainty',
                    'evidence_ids': [], 'question_id': action.frontier_target_id,
                    'status': 'unresolved', 'basis': 'bounded_generated_objection_not_evidence'}
        known = {q.get('id') for q in questions} | {t.get('id') for t in state.theories} | {state.goal, 'goal:G0', 'G0'}
        literature_target = extra.get('theory_id') or theory.get('id')
        if not literature_target and (any(a.get('kind') == 'action_attempt'
                and a.get('action_type') == 'SURVEY' and a.get('status') in {'blocked', 'literature_gap'}
                and a.get('target', target) == target for a in state.artifacts)
                or any(f.get('action_type') == 'SURVEY' and f.get('question_id') == target
                       and f.get('reason') == 'no_verified_literature' for f in state.failed_designs)):
            return None
        if target in known and not any(a.get('kind') == 'literature_survey'
                and (a.get('theory_id') == literature_target if literature_target
                     else a.get('question_id') in {'', target, None}) for a in state.artifacts):
            phase, gap = 'context', 'literature_context_missing'
    elif kind == 'REDESIGN':
        demand = extra.get('demand') or {}
        pending = [r for r in state.design_revisions if r.get('outcome') == 'open'
                   and int(r.get('project_epoch') or 0) == state.project_epoch]
        if demand and not pending and not any(t.get('status') == 'refuted' for t in theories + [theory]):
            phase, gap = 'discrimination', str(demand.get('reason') or demand.get('dimension') or 'design_validity_unresolved')
    elif kind == 'PROJECT_PIVOT':
        if extra.get('trigger') and (state.failed_designs or state.reviews or state.blocked_questions):
            phase, gap = 'feasibility', 'current_direction_infeasible_or_exhausted'
    elif kind == 'THEORIZE':
        parent = next((t for t in state.theories if t.get('id') == extra.get('parent_theory_id')), {})
        if extra.get('revision_id') and parent.get('status') in {'refuted', 'retired'}:
            return None
        if extra.get('discovery_idea_id') or extra.get('revision_id') or (question and not theory):
            phase = 'evidence_acquisition' if extra.get('revision_id') or extra.get('discovery_idea_id') else 'exploration'
            gap = 'hypothesis_not_registered'
    elif kind == 'OBSERVE':
        already_observed = any(e.get('role') == 'QuestionEvidence'
            and e.get('question_id') == question.get('id') and e.get('world_id') == state.world_id
            and e.get('attested') and not e.get('invalidated') for e in state.evidence_records)
        if question and not question.get('evidence_id') and not already_observed:
            phase, gap = 'evidence_acquisition', 'question_observation_missing'
    elif kind == 'SYNTHESIZE':
        if state.evidence_records and state.metrics.get('meaningful_events', 0) > state.metrics.get('last_synthesis_events', 0):
            phase, gap = 'synthesis', 'current_findings_unreported'
    if not gap:
        return None
    return {'id': f'{action.frontier_target_id or target}:{gap}', 'kind': gap,
            'phase': phase, 'question_id': action.frontier_target_id or theory.get('question_id', ''),
            'evidence_ids': basis, 'status': 'unresolved',
            'basis': 'recorded_dependency_not_calibrated_information_gain'}


def _ids(rows: list[Mapping[str, Any]] | list[dict[str, Any]], key: str = "id") -> list[str]:
    out: list[str] = []
    for row in rows or []:
        if isinstance(row, Mapping):
            text = str(row.get(key) or row.get("text") or "").strip()
        else:
            text = str(row).strip()
        if text and text not in out:
            out.append(text)
    return out


def measurement_gaps(state: ScientificState) -> list[dict[str, Any]]:
    """Registered experiments that currently cannot run, and why.

    A requirement whose protocol does not validate used to leave the frontier
    silently: the scheduler skipped it, nothing named the problem, and a project
    asking for five independent samples from a one-sample operator simply had no
    actions and no reason. Naming the shortfall here is not an action — it is
    the record the scheduler, the report and any handoff all read.
    """
    from .discovery import discovery_view
    from .requirement import instrument_gap, protocol_runnable, requirement_of_idea
    from .study import coverage, study_of

    gaps: list[dict[str, Any]] = []
    try:
        ideas = discovery_view(state)["ideas"]
    except (AttributeError, KeyError, TypeError):
        return gaps
    probes = [row for row in getattr(state, "evidence_records", []) or []
              if row.get("role") == "ProbeEvidence"]
    for row in ideas:
        idea_id = str(row.get("id") or "")
        try:
            req = requirement_of_idea(state, row)
            runnable = protocol_runnable(req)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if not runnable.get("runnable"):
            # An idea that has not been operationalised yet is not a refused
            # registration. Discovery's own operators turn a conceptual idea
            # into an experiment; only a protocol the bound world actually
            # rejected needs repairing, and conflating the two makes every
            # early idea look broken.
            reason = str(runnable.get("reason") or "protocol_not_runnable")
            errors = [str(item) for item in runnable.get("errors") or ()]
            unregistered = {"missing_executable_protocol",
                            "world_extension_awaits_admission"}
            entry = {
                "idea_id": idea_id,
                "question_id": str(row.get("question_id") or ""),
                "kind": ("not_yet_operationalized"
                         if unregistered & (set(errors) | {reason})
                         else "registration_gap"),
                "gap": reason,
                "errors": errors,
            }
            # "Not operationalised" is true of every idea before it is compiled,
            # so on its own it diagnoses nothing. Where the reason is that no
            # available instrument reads what the proposal asked for, say which
            # observable was asked for and which ones exist.
            instrument = instrument_gap(state, req)
            if instrument:
                entry["instrument"] = instrument
            gaps.append(entry)
            continue
        if study_of(req.execution) is None:
            continue
        cover = coverage(probes, req.execution, claim_id=idea_id)
        if not cover["complete"]:
            gaps.append({
                "idea_id": idea_id,
                "question_id": str(row.get("question_id") or ""),
                "kind": "coverage_gap",
                "gap": "registered_settings_without_an_admitted_result",
                "errors": list(cover["missing_settings"]),
                "coverage_count": cover["coverage_count"],
                "independent_unit_count": cover["independent_unit_count"],
            })
    return gaps


def frontier_from_state(state: ScientificState) -> dict[str, Any]:
    competing = [
        str(row.get("id") or row.get("text") or "")
        for row in state.theories
        if isinstance(row, Mapping) and row.get("competing")
    ]
    nodes = []
    if state.goal:
        nodes.append({"id": f"goal:{_goal_id(state)}", "kind": "Goal", "status": "active"})
    for row in state.open_questions:
        nodes.append({"id": str(row.get("id") or ""), "kind": "Question", "status": "active"})
    for row in state.resolved_questions:
        nodes.append({"id": str(row.get("id") or ""), "kind": "Question", "status": "resolved"})
    for row in state.blocked_questions:
        nodes.append({"id": str(row.get("id") or ""), "kind": "Question", "status": "blocked"})
    for row in state.theories:
        nodes.append(
            {
                "id": str(row.get("id") or row.get("text") or "")[:80],
                "kind": "Theory",
                "status": str(row.get("status") or "open"),
            }
        )
    return {
        "root_goal": state.goal,
        "active_questions": _ids(state.open_questions),
        "resolved_questions": _ids(state.resolved_questions),
        "blocked_questions": _ids(state.blocked_questions),
        "open_theories": _ids(state.theories),
        "competing_theories": [item for item in competing if item],
        "required_capabilities": list(state.missing_capabilities),
        "measurement_gaps": measurement_gaps(state),
        "verification_debt": int((state.debt or {}).get("verification_debt") or 0),
        "contradictions": [dict(item) for item in state.contradictions],
        "current_frontier_nodes": [item for item in nodes if item.get("status") in {"active", "blocked", "open"}],
        "nodes": nodes,
    }


def _sole_support_row(state: ScientificState, row: Mapping[str, Any]) -> bool:
    from .actions import sole_support

    return sole_support(state, row)


def debt_from_state(state: ScientificState) -> dict[str, int]:
    """Research debt. ``archive_debt`` was removed: it read a metric key
    (``working_archive``) that no code wrote, so it was always zero, and the
    working archive lives outside ScientificState where the reducer cannot see
    it."""
    world_ev = [
        row
        for row in state.evidence_records
        if str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id")
    ]
    unverified = [
        row
        for row in world_ev
        if int(row.get("verify_count") or 0) < 1
        or (_sole_support_row(state, row) and not row.get("replication_of"))
    ]
    stale_worlds = max(0, len(state.world_versions) - 1) if state.world_versions else 0
    unused_caps = [
        name
        for name in state.available_capabilities
        if not any(str(row.get("capability") or "") == name for row in state.evidence_records)
    ]
    orphan_theories = [
        row
        for row in state.theories
        if not row.get("discriminating_predictions")
        or not any(pred.get("competitor_differs") for pred in row.get("discriminating_predictions") or [])
    ]
    return {
        "verification_debt": len(unverified),
        "contradiction_debt": len(state.contradictions),
        "world_staleness": stale_worlds,
        "capability_debt": len(state.missing_capabilities) + len(unused_caps),
        "unresolved_dependency": len(state.blocked_questions)
        + sum(1 for row in state.open_questions if row.get("required_observable") or row.get("required_capability")),
        "orphan_theory_debt": len(orphan_theories),
    }


def graph_from_state(state: ScientificState) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    seen: set[str] = set()

    def node(kind: str, node_id: str, **attrs: Any) -> None:
        key = f"{kind}:{node_id}"
        if not node_id or key in seen:
            return
        seen.add(key)
        nodes.append({"kind": kind, "id": node_id, **attrs})

    def edge(src: str, rel: str, dst: str) -> None:
        if src and rel and dst:
            edges.append({"src": src, "rel": rel, "dst": dst})

    goal_id = f"goal:{_goal_id(state)}"
    if state.goal:
        node("Goal", goal_id, text=state.goal)
    if state.world_id:
        node("World", state.world_id, current=True)
    for row in state.world_versions:
        wid = str(row.get("id") or row.get("world_id") or "")
        node("World", wid)
        parent = str(row.get("parent_id") or "")
        if parent:
            edge(parent, "derived_from", wid)
    for row in state.open_questions + state.resolved_questions + state.blocked_questions:
        qid = str(row.get("id") or "")
        node("Question", qid, status=str(row.get("status") or "open"))
        if state.goal:
            edge(goal_id, "motivates", qid)
        cap = str(row.get("required_capability") or "")
        if cap:
            node("Capability", cap)
            edge(qid, "requires", cap)
        obs = str(row.get("required_observable") or "")
        if obs:
            node("Artifact", obs)
            edge(qid, "requires", obs)
    for row in state.theories:
        tid = str(row.get("id") or row.get("text") or "")[:80]
        node("Theory", tid, epistemic=str(row.get("epistemic") or "GENERATED"))
        qid = str(row.get("question_id") or row.get("target") or "")
        if qid:
            edge(qid, "motivates", tid)
        for pred in row.get("discriminating_predictions") or []:
            maps = str(pred.get("maps_to") or "")
            if maps and maps != "untestable":
                edge(tid, "tests", maps)
    for row in state.evidence_records:
        eid = str(row.get("evidence_id") or "")
        node(
            "Evidence",
            eid,
            epistemic=str(row.get("epistemic") or ""),
            world_id=str(row.get("world_id") or ""),
        )
        qid = str(row.get("question") or row.get("question_id") or row.get("target") or "")
        if qid:
            edge(eid, "answers", qid)
        wid = str(row.get("world_id") or "")
        if wid:
            edge(eid, "valid_in_world", wid)
        tid = str(row.get("theory_id") or "")
        relation = str(row.get("theory_relation") or row.get("verdict") or "")
        if tid and relation in {"supports", "weakens", "contradicts"}:
            edge(eid, relation, tid)
    for name in state.available_capabilities:
        node("Capability", name, status="available")
    for name in state.missing_capabilities:
        node("Capability", name, status="missing")
    return {"nodes": nodes, "edges": edges}


def _goal_id(state: ScientificState) -> str:
    return "G0" if state.goal else ""


def traces_to_root(state: ScientificState, frontier_target_id: str) -> bool:
    """True when the target is the goal, a question, or a node linked from the goal."""
    target = str(frontier_target_id or "").strip()
    if not target:
        return False
    if target == state.goal or target == f"goal:{_goal_id(state)}" or target == "G0":
        return True
    known = set(_ids(state.open_questions + state.resolved_questions + state.blocked_questions))
    known.update(_ids(state.theories))
    known.update(str(row.get("evidence_id") or "") for row in state.evidence_records)
    known.update(state.active_lineages)
    if target in known:
        return True
    graph = state.graph if state.graph.get("edges") else graph_from_state(state)
    goal_id = f"goal:{_goal_id(state)}"
    seen = {goal_id, state.goal}
    queue = [goal_id, state.goal]
    edges = list(graph.get("edges") or [])
    while queue:
        cur = queue.pop(0)
        for edge in edges:
            if edge.get("src") == cur and edge.get("dst") not in seen:
                seen.add(str(edge.get("dst") or ""))
                queue.append(str(edge.get("dst") or ""))
    return target in seen


def blocked_descendants(state: ScientificState, action_type: str, target: str) -> int:
    """How many frontier nodes this action would unlock."""
    if action_type == "ACQUIRE":
        n = 0
        for row in state.open_questions:
            qid = str(row.get("id") or "")
            if target and qid != target and target not in {
                str(row.get("required_observable") or ""),
                str(row.get("required_capability") or ""),
            }:
                continue
            if row.get("required_observable") or row.get("required_capability"):
                n += 1
                n += 1  # OBSERVE
                n += 1  # possible VERIFY
        return n or 1
    if action_type == "EVOLVE_HARNESS":
        return max(1, len(state.missing_capabilities) + len(state.open_questions))
    if action_type == "OBSERVE":
        return 1
    if action_type == "VERIFY":
        return max(1, len(state.contradictions))
    return 0
