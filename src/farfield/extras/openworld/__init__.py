"""Verified open-world research environment.

Production control plane: ScientificState → Frontier → Affordance →
Scheduler → Executor → TrustedKernel → Event → Reducer.

mission.py may bootstrap this environment. It must not own the next action.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from ..generate import GeneratedCard
from ..ideakind import idea_kind_of
from ..lifecycle import ResearchNotebook, load_notebook, save_notebook
from .actions import (
    ACQUIRE,
    ACTION_TYPES,
    ASK,
    PROJECT_PIVOT,
    REDESIGN,
    REVIEW,
    VERIFY,
    EVOLVE_HARNESS,
    FORK_LINEAGE,
    IDEATE,
    OBSERVE,
    PROBE,
    REFLECT,
    SPECS,
    THEORIZE,
    ActionInstance,
    eligible_actions,
)
from .capability import (
    CapabilityRegistry,
    load_capabilities,
    run_capability,
    save_capabilities,
)
from .env import (
    HarnessVersion,
    load_harness,
    project_harness,
    projection_lines,
    save_harness,
)
from .events import EventLog, load_event_log, make_event, save_event_log
from .evolve import (
    HARNESS_FAILURE,
    PATHOLOGY_FLOOR,
    SCIENTIFIC_FAILURE,
    CandidatePatch,
    CreditReport,
    Pathology,
    accumulate_pathology,
    apply_admitted_patch,
    classify_failure,
    evaluate_patch,
    load_pathologies,
    pathology_ready,
    propose_patch,
    save_pathologies,
)
from .executors import DEFAULT_EXECUTORS, maybe_gc, resolve_world_root, fixture_from_root
from .worldio import is_attested
from .frontier import SCIENTIFIC_ACTIONS, traces_to_root
from .kernel import (
    readonly_view,
    MUTABLE_MODULES,
    TRUSTED_KERNEL,
    CandidateResult,
    as_candidate,
    can_promote_to_world,
    make_run_context,
    preserve_evidence_provenance,
    sanitize_evidence_candidate,
    social_consensus_is_not_evidence,
)
from .memory import (
    ArchiveStore,
    add_failure,
    add_harness,
    add_scientific,
    add_working,
    load_archives,
    project_archive,
    projection_token_count,
    save_archives,
)
from .contract import ResearchContract, contract_from_state, root_alignment
from .progress import (
    LANES,
    fork_trajectories,
    judge_branches,
    maybe_stagnation,
    maybe_strategic_stagnation,
    measure_progress,
    merge_branch,
    merge_branches,
    should_terminate,
    update_metrics,
)
from .quality import compute_quality, open_reviews, review_debt
from .resources import budget_phase, budget_state
from .restart import clean_restart_projection
from .reducer import apply_events, rebuild_state, state_digest
from .scheduler import ScoredAction, admissible, schedule, schedule_all
from .state import (
    SCIENTIFIC_STATE_FILE,
    ScientificState,
    load_scientific_state,
    notebook_projection,
    save_scientific_state,
    state_is_empty,
    sync_from_notebook,
)


Executor = Callable[["OpenWorld", ActionInstance], Any]


@dataclass
class OpenWorld:
    workspace: Path | None
    state: ScientificState = field(default_factory=ScientificState)
    harness: HarnessVersion = field(default_factory=lambda: HarnessVersion(version_id="H0"))
    capabilities: CapabilityRegistry = field(default_factory=CapabilityRegistry)
    archives: ArchiveStore = field(default_factory=ArchiveStore)
    pathologies: dict[str, Pathology] = field(default_factory=dict)
    executors: dict[str, Executor] = field(default_factory=dict)
    last_scheduled: ScoredAction | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    log: EventLog = field(default_factory=EventLog)
    workers_connected: bool = False
    # Set by `load` when a stored snapshot disagreed with its own event log.
    # Outside canonical state on purpose: recording it inside would change the
    # rebuilt state's digest and break the equality being checked.
    replay_divergence: dict[str, Any] | None = None
    # True once an adapter that performs IDEATE is connected. The discovery
    # runtime offers its moves only then, so a controller wired for evidence
    # work alone schedules exactly as it always did.
    discovery_connected: bool = False

    def connect_workers(self, adapter: Any) -> None:
        """Bind worker services, never their own orchestration loop or state writer."""
        from copy import deepcopy
        import time

        for kind, worker in adapter.executors().items():
            def invoke(env: "OpenWorld", action: ActionInstance, worker: Any = worker) -> Any:
                if action.action_type == "VERIFY":
                    target = next((r for r in env.state.evidence_records
                                   if r.get("evidence_id") == action.target), {})
                    if (action.extra.get("evidence_role") == "QuestionEvidence"
                            or (target.get("registration") or {}).get("version") in {"computational-protocol-v2", "computational-protocol-v3"}):
                        return DEFAULT_EXECUTORS["VERIFY"](env, action)
                ledger = getattr(getattr(adapter, "client", None), "ledger", None)
                before_tokens = float(getattr(ledger, "spent_tokens", 0))
                before_calls = float(getattr(ledger, "spent_calls", 0))
                started = time.monotonic()
                result = as_candidate(worker(action, deepcopy(env.state.to_dict()), workspace=env.workspace,
                              world_id=env.state.world_id, projection=env.project_for(action)))
                # Calls, not actions. One SURVEY can extract a dozen papers, and
                # the resource view billing it as a single api_call disagreed
                # with the ledger that actually enforces the envelope — two
                # meters of the same name, an order of magnitude apart.
                cost = {"tokens": float(getattr(ledger, "spent_tokens", 0)) - before_tokens,
                        "api_calls": float(getattr(ledger, "spent_calls", 0)) - before_calls,
                        "wall_seconds": time.monotonic() - started}
                result.extras["measured_cost"] = cost
                if result.evidence is not None:
                    result.evidence["measured_cost"] = cost
                result.events = [replace(event, payload={**event.payload, "measured_cost": {
                    k: float((event.payload.get("measured_cost") or {}).get(k) or 0) + cost[k] for k in cost}})
                    if event.event_type == "EvidenceAdded" else event for event in result.events]
                return result
            self.executors[kind] = invoke
        self.workers_connected = True
        self.discovery_connected = self.discovery_connected or IDEATE in self.executors

    @classmethod
    def load(cls, workspace: Path | None) -> "OpenWorld":
        env = cls(workspace=workspace)
        env.state = load_scientific_state(workspace)
        env.harness = load_harness(workspace)
        env.capabilities = load_capabilities(workspace)
        env.archives = load_archives(workspace)
        env.pathologies = load_pathologies(workspace)
        env.log = load_event_log(workspace)
        notebook = load_notebook(workspace)
        if state_is_empty(env.state) and notebook is not None:
            sync_from_notebook(env.state, notebook)
        # The log is the only writer, so on a resume it is also the authority.
        # A stored state that disagrees with its own log is not a tie to break
        # by preference: it means the saved snapshot was produced by different
        # derivation code, and continuing from it would carry a state the log
        # cannot account for. Rebuild, and record that it happened rather than
        # doing it silently.
        if env.log.events:
            from .reducer import rebuild_state, state_digest

            replayed = rebuild_state(env.log)
            loaded = state_digest(env.state)
            if loaded != state_digest(replayed):
                # The record lives on the environment, not in the state. Writing
                # it into canonical state would change the rebuilt state's own
                # digest, so the state would no longer equal its replay — the
                # invariant this check exists to protect.
                env.replay_divergence = {
                    "rule": "replay-divergence-v1",
                    "loaded_digest": loaded,
                    "replayed_digest": state_digest(replayed),
                    "events": len(env.log.events),
                    "resolution": "rebuilt_from_the_event_log",
                    "cause": "a snapshot saved by different derivation code; the "
                             "log is the only writer, so it is the authority",
                }
                env.state = replayed
        env.state.harness_version = env.harness.version_id
        return env

    def persist(self) -> None:
        self.state.harness_version = self.harness.version_id
        # The log first. It is the only writer and the authority on resume, so a
        # crash between the two writes leaves a log ahead of its snapshot — which
        # a load detects and rebuilds from — never a snapshot the log cannot
        # account for.
        save_event_log(self.workspace, self.log)
        save_scientific_state(self.workspace, self.state)
        save_harness(self.workspace, self.harness)
        save_capabilities(self.workspace, self.capabilities)
        save_archives(self.workspace, self.archives)
        save_pathologies(self.workspace, self.pathologies)
        if self.workspace is not None:
            save_notebook(
                self.workspace,
                ResearchNotebook.from_dict(notebook_projection(self.state)),
            )

    def eligible(self, *, probe_ready: Mapping[str, Any] | None = None) -> list[ActionInstance]:
        strategic = maybe_strategic_stagnation(self.state)
        # A busy-but-stuck project should widen its options, not only an idle one.
        stagnant = maybe_stagnation(self.state) is not None or strategic is not None
        # Persisted state may name an older, now revoked capability. Runtime
        # registry decisions override those names without rewriting history.
        usable = self.capabilities.usable_names()
        unavailable = set(self.capabilities.names()) - set(usable)
        view = replace(self.state, available_capabilities=list(dict.fromkeys([
            name for name in self.state.available_capabilities if name not in unavailable] + list(usable))))
        return eligible_actions(
            view,
            pathology_ready=bool(pathology_ready(self.pathologies)),
            stagnation=stagnant,
            probe_ready=probe_ready,
            workers_connected=self.workers_connected,
            strategic_stagnation=strategic,
            discovery=self.discovery_connected,
        )

    def schedule(self, actions: list[ActionInstance] | None = None) -> ScoredAction | None:
        pool = actions if actions is not None else self.eligible()
        chosen = schedule(self.state, pool)
        self.last_scheduled = chosen
        return chosen

    def project_for(self, action: ActionInstance) -> dict[str, Any]:
        """The context this action should see — not the whole trajectory.

        The projection carries the contract, the derived decision journal, the
        evidence relevant to this action's target, the standing objections and
        the remaining budget. After a pivot it carries the clean restart instead
        of the retired direction's narrative, so an abandoned line cannot anchor
        the work that replaces it.
        """
        from .journal import contingency_plan, research_journal

        overlay = project_harness(
            self.harness,
            action_type=action.action_type,
            lineage_id=action.lineage_id,
        )
        archive = project_archive(
            self.archives,
            action_type=action.action_type,
            world_id=self.state.world_id,
        )
        theory_id = str(action.extra.get("theory_id") or action.target or "")
        journal = research_journal(
            self.state,
            question_id=str(action.frontier_target_id or ""),
            theory_id=theory_id,
        )
        payload = {
            "harness_lines": projection_lines(
                self.harness, action_type=action.action_type, lineage_id=action.lineage_id
            ),
            "archive": archive,
            "archive_tokens": projection_token_count(archive),
            "overlay_names": [item.name for item in overlay],
            "research_journal": journal,
            "research_contract": journal["research_contract"],
            "open_objections": journal["open_objections"],
            "relevant_world_evidence": journal["relevant_world_evidence"],
            "remaining_resources": journal["resources"],
        }
        if action.action_type in {PROBE, VERIFY}:
            payload["contingency_plan"] = contingency_plan(self.state, action)
        if int(self.state.project_epoch or 0):
            restart = next(
                (row for row in reversed(self.state.artifacts)
                 if row.get("kind") == "clean_restart_projection"), None)
            if restart is not None:
                payload["clean_restart"] = restart
        payload["projection_tokens"] = projection_token_count(payload)
        return payload

    def note_harness_failure(self, *, where: str, why: str, example: str = "") -> Pathology:
        row = accumulate_pathology(self.pathologies, where=where, why=why, example=example)
        add_failure(
            self.archives,
            {
                "class": HARNESS_FAILURE,
                "where": where,
                "why": why,
                "example": example,
                "count": row.count,
            },
        )
        return row

    def note_scientific_failure(self, *, why: str, example: str = "") -> None:
        add_failure(
            self.archives,
            {"class": SCIENTIFIC_FAILURE, "why": why, "example": example},
        )

    def _ensure_baseline(self) -> None:
        if not self.log.baseline:
            self.log.baseline = self.state.to_dict()

    def _admit(self, action: ActionInstance, candidate: CandidateResult) -> CandidateResult:
        """TrustedKernel mutation boundary. Executors cannot write WORLD or state.

        An executor proposes; this decides what enters the log:

        - only the event types its action declares (`ActionSpec.emits`),
          artifacts only of the kinds it declares, and a report that it was
          blocked or failed — blocking only its own target;
        - evidence only with an identity, WORLD only against a matching attested
          freeze, and a probe row judged here by the validity contract rather
          than by the worker's own columns;
        - a resolution only when it names, by id, admitted evidence the kernel
          found able to corroborate.

        A proposal that crosses the boundary is named in the result and counted
        as an integrity gap; a resolution the evidence does not carry is named
        too, but it is not a breach — the kernel simply does not credit it.
        """
        from ..validity import assess_probe, sample_floor
        from .actions import ALWAYS_ADMISSIBLE

        refused: list[str] = []
        limits = dict(self.state.contract or {}).get("resource_limits")
        floor = sample_floor(limits if isinstance(limits, Mapping) else None)
        allowed = set(action.spec.emits) | set(ALWAYS_ADMISSIBLE)
        own_targets = {str(action.target or ""), str(action.frontier_target_id or "")} - {""}
        open_ids = {str(row.get("id") or "") for row in self.state.open_questions}

        def checked(row: Mapping[str, Any]) -> dict[str, Any]:
            payload = dict(row)
            if payload.get('study_stage') in {
                    'pilot', 'calibration', 'scientific_pilot', 'confirmatory', 'replication'
                    } and payload.get('discovery_idea_id'):
                from .experiment import admission_errors
                errors = admission_errors(payload, self.workspace, self.state)
                if str(payload.get('question_id') or '') != str(action.frontier_target_id or action.target or ''):
                    errors.append('experiment_question_not_action_target')
                if errors:
                    payload.update(epistemic='GENERATED', attested=False,
                                   validation_gaps=list(payload.get('validation_gaps') or []) + errors,
                                   promotion_refused='; '.join(errors))
            generated_origin = any(str(payload.get(k) or "").upper() in {"GENERATED", "SYNTHETIC", "PLACEBO", "E0_SYNTHETIC"}
                for k in ("kind", "probe_kind", "world_role", "provenance", "world_provenance"))
            if generated_origin:
                payload.update(epistemic="GENERATED", attested=False,
                               promotion_refused="generated provenance cannot be relabelled WORLD")
            if payload.get("epistemic") == "WORLD":
                wid = str(payload.get("world_id") or "")
                root = resolve_world_root(self.workspace, wid)
                fixture = fixture_from_root(root, wid) if root else None
                if not (fixture and is_attested(fixture)
                        and (payload.get("world_digest") or payload.get("digest")) == fixture.digest):
                    payload.update(epistemic="GENERATED", attested=False,
                                   promotion_refused="WORLD evidence needs a matching attested freeze digest")
            sanitized, _ = sanitize_evidence_candidate(payload)
            payload = sanitized or payload
            if str(payload.get("role") or "") == "ProbeEvidence":
                payload.update(assess_probe(payload, min_sample=floor).fields(payload))
            if str(payload.get("study_stage") or "") in {"pilot", "calibration", "scientific_pilot"}:
                # A trusted observation may remain WORLD. It cannot promote a claim.
                payload["promotion_ok"] = False
                payload["cannot_corroborate"] = True
                payload["corroborates"] = False
                validity = payload.get("validity")
                if isinstance(validity, Mapping):
                    payload["validity"] = {**validity, "corroborates": False}
            return payload

        def boundary(event: Any) -> str:
            """Why this proposed event may not enter the log, or "" to admit it."""
            kind = event.event_type
            payload = dict(event.payload or {})
            if kind == "ArtifactAdded":
                artifact = str(payload.get("kind") or "")
                if artifact not in action.spec.artifact_kinds:
                    return (f"ArtifactAdded: {action.action_type} does not produce "
                            f"{artifact or 'unnamed'} artifacts")
                return ""
            if kind not in allowed:
                return f"{kind}: not an event {action.action_type} may propose"
            if kind == "EvidenceAdded" and not str(payload.get("evidence_id") or ""):
                return "EvidenceAdded: evidence without an EvidenceID"
            if kind == 'EvidenceRejected' and action.action_type == VERIFY:
                if not action.target or payload.get('evidence_id') != action.target:
                    return 'EvidenceRejected: verifier may only invalidate its target'
            if kind == "ActionBlocked":
                named = str(payload.get("question_id") or payload.get("target") or "")
                if named in open_ids and named not in own_targets:
                    return f"ActionBlocked: question {named} is not this action's target"
            return ""

        def regraded(event: Any) -> Any:
            """A proposed proposition relation is recomputed, never believed.

            A worker may write ``claim_relation`` on anything it emits. Whatever
            it says there, what enters the log is this kernel's own derivation
            over both sentences' structure, carrying the rule version that
            produced it. An unstructurable pair stays ``unknown``.
            """
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            if not isinstance(payload.get('claim_relation'), Mapping):
                return event
            from .claim_relations import validate_claim_relation
            return replace(event, payload={
                **dict(payload),
                'claim_relation': validate_claim_relation(payload['claim_relation'], self.state)})

        world = None
        events = []
        for event in candidate.events or []:
            event = regraded(event)
            why = boundary(event)
            if why:
                refused.append(why)
                continue
            if event.event_type == "EvidenceAdded":
                payload = checked(event.payload)
                from .experiment import computational_theory
                theory = computational_theory(payload)
                if theory and not any(t.get('id') == theory['id'] for t in self.state.theories) and not any(
                        e.event_type == 'TheoryCreated' and e.payload.get('id') == theory['id'] for e in events):
                    events.append(make_event('TheoryCreated', theory, action_type=action.action_type))
                events.append(replace(event, payload=payload))
                continue
            if event.event_type == 'ReviewRecorded':
                from .judgment import admission_error
                why = admission_error(event.payload or {}, self.state)
                if why:
                    refused.append(why)
                    continue
                events.append(replace(event, payload={**dict(event.payload), 'project_epoch': self.state.project_epoch}))
                continue
            from .discovery import DISCOVERY_ARTIFACT_KINDS, stamp_generated

            # A discovery record is GENERATED whatever it says about itself.
            if (event.event_type == "ArtifactAdded"
                    and str((event.payload or {}).get("kind") or "") in DISCOVERY_ARTIFACT_KINDS):
                payload = stamp_generated(event.payload)
                if payload.get('kind') == 'discovery_decision':
                    from .response import result_packet, validate_response, digest
                    parent = payload.get('parent_id')
                    if (not isinstance(parent, str) or not parent
                            or parent != action.extra.get('parent_idea_id')
                            or not any(r.get('kind') == 'discovery_idea' and r.get('id') == parent
                                       for r in self.state.artifacts)):
                        refused.append('discovery_decision_parent_not_action_parent')
                        continue
                    packet = result_packet(self.state, parent)
                    response = payload.get('result_response')
                    why = validate_response(packet, response)
                    if why or not packet['required'] or response.get('disposition') != 'pause':
                        refused.append(why or 'discovery_decision_requires_pause_of_observed_idea')
                        continue
                    payload = {'kind': 'discovery_decision', 'parent_id': parent,
                               'id': 'decision-' + digest(response)[:20],
                               'project_epoch': self.state.project_epoch, 'result_response': response,
                               'reference_check': 'coverage_and_freshness_not_semantic_validity',
                               'epistemic': 'GENERATED'}
                if payload.get('kind') == 'discovery_idea':
                    if (not isinstance(payload.get('id'), str) or not payload['id'].strip()
                            or not isinstance(payload.get('idea'), Mapping)
                            or any(key in payload and not isinstance(payload[key], str)
                                   for key in ('question_id', 'parent_id'))
                            or ('experiment' in payload['idea']
                                and not isinstance(payload['idea']['experiment'], Mapping))):
                        refused.append('discovery_idea_invalid_structure')
                        continue
                    existing_ideas = {str(a.get('id') or ''): a for a in self.state.artifacts
                                      if a.get('kind') == 'discovery_idea'}
                    pending_ideas = {e.payload.get('id') for e in events
                                     if e.event_type == 'ArtifactAdded' and e.payload.get('kind') == 'discovery_idea'}
                    if payload['id'] in existing_ideas or payload['id'] in pending_ideas:
                        refused.append('discovery_idea_identity_is_immutable')
                        continue
                    parent = str(payload.get('parent_id') or '')
                    expected_parent = str(action.extra.get('parent_idea_id') or '')
                    if ((parent and (parent == payload['id'] or parent not in existing_ideas))
                            or (expected_parent and parent != expected_parent)):
                        refused.append('discovery_idea_parent_not_known_or_not_action_parent')
                        continue
                    from .response import result_packet, validate_response
                    why = validate_response(result_packet(self.state, parent), payload.get('result_response'))
                    if why or (payload.get('result_response') or {}).get('disposition') == 'pause':
                        refused.append(why or 'paused_decision_cannot_propose_an_idea')
                        continue
                    if payload.get('result_response'):
                        payload['reference_check'] = 'coverage_and_freshness_not_semantic_validity'
                    questions = {str(q.get('id') or ''): q for q in (
                        self.state.open_questions + self.state.resolved_questions
                        + self.state.partial_questions + self.state.blocked_questions)}
                    parent_question = str(existing_ideas.get(parent, {}).get('question_id') or '')
                    action_question = str(action.extra.get('question_id') or '')
                    if ((parent_question and parent_question not in questions)
                            or (action_question and action_question not in questions)):
                        refused.append('discovery_idea_context_question_unknown')
                        continue
                    action_question = action_question or next((qid for qid in (
                        action.frontier_target_id, action.target) if qid in questions), '')
                    context_question = action_question or parent_question
                    if not context_question and len(questions) == 1:
                        context_question = next(iter(questions))
                    if (not context_question or (parent_question and parent_question != context_question)):
                        refused.append('discovery_idea_question_lineage_mismatch')
                        continue
                    proposed_question = payload.get('question_id') or context_question
                    transition = payload.get('question_transition')
                    if 'question_transition' in payload:
                        from .contract import validate_question_transition
                        if (not isinstance(transition, Mapping)
                                or any(not isinstance(transition.get(key), str) or not transition[key].strip()
                                       for key in ('id', 'parent_question_id'))
                                or transition['parent_question_id'] != context_question
                                or proposed_question not in {context_question, transition['id']}
                                or any(e.event_type == 'QuestionCreated' and e.payload.get('id') == transition['id']
                                       for e in events)):
                            refused.append('question_transition:invalid_structure_or_parent_binding')
                            continue
                        ok, why = validate_question_transition(self.state, transition)
                        if not ok:
                            refused.append('question_transition:' + why)
                            continue
                        events.append(make_event('QuestionCreated', transition, action_type=action.action_type))
                        payload['question_id'] = transition['id']
                    elif proposed_question != context_question or proposed_question not in questions:
                        refused.append('discovery_idea_question_change_requires_transition')
                        continue
                    else:
                        payload['question_id'] = proposed_question
                    observed = [r for r in self.state.evidence_records if parent and
                                r.get('discovery_idea_id') == parent and r.get('epistemic') == 'WORLD'
                                and r.get('attested') and r.get('execution_status') == 'ran']
                    if observed:
                        current_experiment = (payload.get('idea') or {}).get('experiment') or {}
                        changed = any(current_experiment != (r.get('registration') or {}).get('execution')
                                      for r in observed)
                        payload['experiment_lineage'] = {
                            'parent_idea_id': parent,
                            'parent_evidence_ids': [r['evidence_id'] for r in observed],
                            'parent_registration_digests': sorted({str(r.get('registration_digest') or '') for r in observed}),
                            'relationship': 'new_experiment' if changed else 'new_hypothesis_same_experiment',
                            'satisfies_parent_prediction': False,
                        }
                events.append(replace(event, payload=payload))
            else:
                events.append(event)
        evidence = candidate.evidence
        if evidence is not None and "EvidenceAdded" not in allowed:
            refused.append(f"evidence: {action.action_type} does not produce evidence")
            evidence = None
        elif evidence is not None and not str(evidence.get("evidence_id") or ""):
            refused.append("evidence: evidence without an EvidenceID")
            evidence = None
        evidence = checked(evidence) if evidence is not None else None
        gap = str((evidence or {}).get("promotion_refused") or "")
        if evidence is not None:
            ok, gaps = can_promote_to_world(evidence, world=world)
            evidence["promotion_ok"] = ok
            evidence["promotion_gaps"] = list(gaps)
            if str(evidence.get("study_stage") or "") in {"pilot", "calibration", "scientific_pilot"}:
                evidence["promotion_ok"] = False
                evidence["cannot_corroborate"] = True
            elif not ok and str(evidence.get("epistemic") or "") == "WORLD":
                evidence["epistemic"] = "GENERATED"
                evidence["attested"] = False
                gap = gap or "; ".join(gaps)
            already = any(
                getattr(item, "event_type", "") == "EvidenceAdded"
                and dict(getattr(item, "payload", {})).get("evidence_id") == evidence.get("evidence_id")
                for item in events
            )
            if not already:
                events.append(
                    make_event(
                        "EvidenceAdded",
                        evidence,
                        action_type=action.action_type,
                        frontier_target_id=action.frontier_target_id or action.target,
                    )
                )
        admitted_evidence = [event.payload for event in events if event.event_type == "EvidenceAdded"]
        # An empty id is no id. A row admitted without one put None into this
        # set, and a resolution naming no evidence then matched it.
        valid_evidence = {str(row.get("evidence_id")): row for row in admitted_evidence
                          if row.get("evidence_id") and row.get("promotion_ok")
                          and not row.get("cannot_corroborate") and not row.get("invalidated")}
        canonical_evidence = {str(row.get('evidence_id') or ''): row for row in self.state.evidence_records}
        theory_questions = {str(row.get('id') or ''): str(row.get('question_id') or '')
                            for row in self.state.theories}
        for item in events:
            if item.event_type == 'TheoryCreated':
                theory_questions.setdefault(str(item.payload.get('id') or ''),
                                            str(item.payload.get('question_id') or ''))
        known_questions = {str(q.get('id') or '') for q in (self.state.open_questions
                           + self.state.partial_questions + self.state.blocked_questions)}

        def resolution_is_bound(event: Any) -> bool:
            payload = event.payload
            qid = payload.get('id') or payload.get('question_id')
            eid = payload.get('evidence_id')
            if not isinstance(qid, str) or not isinstance(eid, str) or qid not in known_questions:
                return False
            row = valid_evidence.get(eid)
            if not row or row.get('question_id') != qid:
                return False
            tid = row.get('theory_id')
            if tid and (not isinstance(tid, str) or theory_questions.get(tid) != qid):
                return False
            if payload.get('theory_id') and payload['theory_id'] != tid:
                return False
            if row.get('role') == 'ProbeEvidence' and not tid:
                return False
            existing = canonical_evidence.get(eid)
            if existing and (existing.get('invalidated') or any(
                    existing.get(key) != row.get(key) for key in ('question_id', 'theory_id'))):
                return False
            if any(other.get('evidence_id') == eid and any(other.get(key) != row.get(key)
                    for key in ('question_id', 'theory_id')) for other in admitted_evidence):
                return False
            if any(item.event_type == 'EvidenceRejected' and item.payload.get('evidence_id') == eid
                   for item in events):
                return False
            return True
        resolutions = {"QuestionResolved", "QuestionPartiallyResolved"}
        uncredited = [f"{event.event_type} {event.payload.get('id') or ''}: rests on no admitted "
                      "evidence the kernel credits"
                      for event in events
                      if event.event_type in resolutions
                      and not resolution_is_bound(event)]
        events = [event for event in events
                  if event.event_type not in resolutions
                  or resolution_is_bound(event)]
        return CandidateResult(
            status=candidate.status,
            events=events,
            evidence=evidence,
            world_id=candidate.world_id or self.state.world_id,
            evidence_id=candidate.evidence_id or str((evidence or {}).get("evidence_id") or ""),
            unlocked=candidate.unlocked,
            integrity_gap=gap or candidate.integrity_gap or "; ".join(refused),
            harness_general=candidate.harness_general,
            extras={**dict(candidate.extras),
                    **({"refused_events": refused} if refused else {}),
                    **({"resolutions_not_admitted": uncredited} if uncredited else {})},
        )

    def execute(self, action: ActionInstance) -> dict[str, Any]:
        before = ScientificState.from_dict(dict(readonly_view(self.state)))
        self._ensure_baseline()
        commitment = action.extra.get("branch_commit")
        if commitment and action.extra.get("study_stage") == "confirmatory":
            idea_id = str(action.extra.get("discovery_idea_id") or "")
            branch = next((b for b in self.state.research_branches if b.get("idea_id") == idea_id), None)
            if branch and not branch.get("commit"):
                # Persist before the experiment: a result cannot choose its
                # own stopping rule or retrospectively rewrite the commitment.
                self.state = apply_events(self.state, [make_event("DecisionRecorded", {
                    "kind": "branch_commit", "branch_id": branch["id"],
                    "idea_id": idea_id, "commit": dict(commitment),
                }, action_type=action.action_type)], log=self.log)
                self.persist()
        projection = self.project_for(action)
        frontier_id = action.frontier_target_id or action.target or self.state.goal
        from .response import action_pause_reason
        from .scheduler import tool_policy_reason
        pause = tool_policy_reason(self.state, action) or action_pause_reason(self.state, action)
        if pause:
            candidate = CandidateResult(status='blocked', extras={'reason': pause})
        elif action.action_type in SCIENTIFIC_ACTIONS and not frontier_id:
            candidate = CandidateResult(
                status="blocked",
                events=[
                    make_event(
                        "ActionBlocked",
                        {"reason": "no_frontier_target"},
                        action_type=action.action_type,
                    )
                ],
                extras={"reason": "no_frontier_target"},
            )
        else:
            default_executor = False
            fn = self.executors.get(action.action_type)
            if fn is None:
                fn = self.executors.get(action.spec.executor)
            if fn is None:
                fn = DEFAULT_EXECUTORS.get(action.action_type)
                default_executor = fn is not None
            if fn is None:
                # "recorded" reads as success. An action nothing can perform
                # should say that, so a spec added without an executor fails
                # loudly instead of appearing to do its job.
                raw: Any = {"status": "blocked", "action": action.action_type,
                            "reason": "no_executor_for_this_action"}
            else:
                raw = fn(self, action)
            if action.action_type == EVOLVE_HARNESS and not callable(
                self.executors.get(EVOLVE_HARNESS)
            ):
                raw = self.evolve()
            candidate = as_candidate(raw)
            if default_executor and 'measured_cost' not in candidate.extras:
                # Built-in deterministic executors make no model requests.
                # Unknown third-party workers retain the legacy accounting floor.
                candidate.extras['measured_cost'] = {'api_calls': 0.0, 'tokens': 0.0}
        admitted = self._admit(action, candidate)
        if admitted.evidence_id or admitted.extras.get("pilot_verdict"):
            from .discovery import record_decision_outcome
            admitted.events.append(make_event("DecisionRecorded", record_decision_outcome(
                before, expected=str(action.extra.get("expected_result") or ""),
                actual=str(admitted.extras.get("pilot_verdict") or admitted.status or ""),
                branch_id=str(action.extra.get("branch_id") or ""),
                decision=action.action_type), action_type=action.action_type))
        if self.workers_connected:
            from .actions import _action_variant, action_input_key
            from .recovery import recovery_key, stall_key
            admitted.events.append(make_event("ArtifactAdded", {
                "kind": "action_attempt", "action_type": action.action_type,
                "project_epoch": int(before.project_epoch),
                "target": action.target, "status": admitted.status,
                "input_key": action_input_key(before, action),
                # The gap this attempt was trying to remove, named so a rewrite
                # of the question cannot buy the same attempt a second budget.
                "recovery_key": recovery_key(before, action),
                # The situation this turn faced, so a turn that keeps coming
                # back to the same unchanged situation can be recognised as
                # such however differently its inputs were worded.
                "stall_key": stall_key(before, action),
                # A revision the judgment panel asked for. Counted, because a
                # panel that is never satisfied is otherwise an unbounded spend.
                "judged_below_the_bar": bool(action.extra.get("judged_below_the_bar")),
                # What made this attempt distinct from its siblings, so a
                # repeated failure can be diagnosed without replaying the log.
                "variant": _action_variant(action),
                "research_lane": action.extra.get("research_lane", ""),
                "study_stage": action.extra.get("study_stage", ""),
                "inquiry_lineage_id": (action.extra.get("knowledge_request") or {}).get("lineage_id", ""),
                # How much this attempt actually changed, so a repeatedly inert
                # action type can be recognised within the run rather than
                # rediscovered every tick.
                "events_admitted": len([
                    e for e in admitted.events
                    if e.event_type not in {"ActionBlocked", "ActionFailed"}]),
                "reason": admitted.extras.get("reason", ""),
                # What this attempt actually came back with. Two identical
                # outcomes from the same action on the same target is a fact the
                # controller has already learned.
                "outcome_key": str(
                    admitted.extras.get("reason")
                    or admitted.extras.get("outcome")
                    or admitted.integrity_gap
                    or admitted.status or ""),
                "theory_id": action.extra.get("theory_id", ""),
                "question_id": action.extra.get("question_id", ""),
                "discovery_idea_id": action.extra.get("discovery_idea_id", ""),
                "evidence_id": admitted.evidence_id,
                "measured_cost": admitted.extras.get("measured_cost", {}),
                # A model service that did not answer, and whether its balance is
                # spent: the resource view tells weather from an empty account.
                "service_unavailable": (str(admitted.extras.get("service") or "")
                                        if admitted.status == "service_unavailable" else ""),
                "quota_exhausted": bool(admitted.status == "service_unavailable"
                                        and admitted.extras.get("quota_exhausted")),
                "budget_meter": (str(admitted.extras.get('budget_meter') or '')
                                 if admitted.status == 'budget_exhausted' else ''),
            }, action_type=action.action_type))
        preview = apply_events(self.state, admitted.events)
        progress = measure_progress(before, preview)
        tick_event = make_event(
            "TickAdvanced",
            {
                "meaningful": progress.meaningful,
                # Operational progress and research progress are different
                # claims and drive different clocks.
                "research_meaningful": progress.research_meaningful,
                "central_question_progress": progress.central_question_progress,
                "decision_relevant_evidence": progress.decision_relevant_evidence,
                "quality_gap_reduction": progress.quality_gap_reduction,
                "critical_review_resolution": progress.critical_review_resolution,
                "peripheral_question_progress": progress.peripheral_question_progress,
                "caveat_only_repair": progress.caveat_only_repair,
                "same_failure_repetition": progress.same_failure_repetition,
                "low_value_activity": progress.low_value_activity,
                "successful_reorientation": progress.successful_reorientation,
                "replication_progress": progress.replication_progress,
                "quality_dimension_progress": progress.quality_dimension_progress,
                "objection_discovery": progress.objection_discovery,
                "failure_information": progress.failure_information,
                "action_type": action.action_type,
                "frontier_target_id": frontier_id,
                "scripted": False,
                "unlock": admitted.unlocked,
                "integrity_gap": admitted.integrity_gap,
                "archive_hit": bool(
                    projection["archive"].get("scientific") or projection["archive"].get("harness")
                ),
                "harness_general": admitted.harness_general,
                "goal_drift": bool(
                    action.action_type in SCIENTIFIC_ACTIONS
                    and frontier_id
                    and not traces_to_root(before, frontier_id)
                ),
            },
            action_type=action.action_type,
            frontier_target_id=frontier_id,
        )
        self.state = apply_events(
            self.state,
            list(admitted.events) + [tick_event],
            log=self.log,
            action=action,
        )
        context = make_run_context(
            world_id=str(admitted.world_id or self.state.world_id),
            evidence_id=str(admitted.evidence_id or ""),
            harness_version=self.harness.version_id,
            capability_versions={
                name: self.capabilities.items[name].version
                for name in self.capabilities.names()
                if name in self.capabilities.items
            },
        )
        result = admitted.to_dict()
        result.setdefault("action_type", action.action_type)
        result.setdefault("target", action.target)
        result["frontier_target_id"] = frontier_id
        result["projection"] = {
            "overlay_names": projection["overlay_names"],
            "archive_tokens": projection["archive_tokens"],
            "projection_tokens": projection.get("projection_tokens"),
            "harness_lines": list(projection["harness_lines"]),
            "objections_shown": [row.get("review_id") for row in projection["open_objections"]],
            "next_decisive_action": projection["research_journal"]["next_decisive_action"],
        }
        result["run_context"] = context.to_dict()
        result["harness_version"] = self.harness.version_id
        result["world_id"] = self.state.world_id or admitted.world_id
        if admitted.evidence:
            evidence = dict(admitted.evidence)
            evidence.setdefault("run_context", context.to_dict())
            evidence.setdefault("harness_version", self.harness.version_id)
            if evidence.get("promotion_ok"):
                add_scientific(self.archives, evidence)
            else:
                add_working(self.archives, evidence)
            result["evidence"] = evidence
        maybe_gc(self)
        self.history.append(result)
        self.persist()
        return result

    def evolve(self, patch: CandidatePatch | None = None, **gate: Any) -> dict[str, Any]:
        """Record research-time proposals; admission belongs after mission settlement.

        Legacy caller-supplied cases are accepted for API compatibility, but
        cannot authorize installation during scientific research.
        """
        ready = pathology_ready(self.pathologies)
        if patch is None:
            if not ready:
                return {"status": "no_pathology"}
            patch = propose_patch(ready[0])
        parent_world = self.state.world_id
        parent_harness = self.harness.version_id
        reason = "research_policy_requires_completed_mission_and_heldout_receipt"
        events = [
            make_event(
                "HarnessPatchProposed",
                {
                    "pathology": patch.pathology,
                    "module": patch.module,
                    "status": "proposed",
                    "reason": reason,
                },
                action_type=EVOLVE_HARNESS,
            )
        ]
        add_harness(
            self.archives,
            {
                "patch": patch.to_dict(),
                "status": "proposed",
                "reason": reason,
                "harness_version": parent_harness,
                "parent": parent_harness,
                "world_unchanged": True,
            },
        )
        self._ensure_baseline()
        self.state = apply_events(self.state, events, log=self.log)
        return {
            "status": "proposed",
            "reason": reason,
            "harness_version": self.harness.version_id,
            "parent_harness": parent_harness,
            "world_id": self.state.world_id,
            "world_unchanged": self.state.world_id == parent_world,
            "capability": patch.capability.name if patch.capability else "",
            "unlocked": False,
            "harness_general": False,
            "events_applied": True,
        }

    def tick(self, *, probe_ready: Mapping[str, Any] | None = None) -> dict[str, Any]:
        actions = self.eligible(probe_ready=probe_ready)
        _, refused = admissible(self.state, actions)
        chosen = self.schedule(actions)
        event = {
            "stage": "openworld_tick",
            "controller": "OpenWorld",
            "eligible": [item.to_dict() for item in actions],
            "eligible_types": [item.action_type for item in actions],
            # An eligible action that the scheduler refuses must say why, or a
            # whole action type can vanish from a run unnoticed.
            "refused": refused,
            "stagnation": None
            if maybe_stagnation(self.state) is None
            else maybe_stagnation(self.state).to_dict(),
            "strategic_stagnation": None
            if maybe_strategic_stagnation(self.state) is None
            else maybe_strategic_stagnation(self.state).to_dict(),
        }
        if chosen is None:
            event["scheduled"] = None
            return event
        event["scheduled"] = chosen.to_dict()
        from .discovery import reason
        judgment = reason(self.state)
        event["reason"] = {
            "mode": judgment.get("mode"),
            "trigger": judgment.get("trigger"),
            "next_uncertainty": judgment.get("next_uncertainty"),
            "basis": judgment.get("basis"),
        }
        event["research_frontier"] = [
            {"id": row.get("id"), "status": row.get("status"),
             "mechanism": row.get("mechanism"),
             "decisive_uncertainty": row.get("decisive_uncertainty")}
            for row in judgment.get("branches") or []
        ]
        result = self.execute(chosen.action)
        event["result"] = result
        event["metrics"] = dict(self.state.metrics)
        event["harness_version"] = self.harness.version_id
        event["world_id"] = self.state.world_id
        event["frontier"] = dict(self.state.frontier)
        event["debt"] = dict(self.state.debt)
        event["quality"] = dict(self.state.quality)
        event["objective"] = dict(self.state.objective)
        from .exploration import trajectory_metrics

        event["trajectory"] = trajectory_metrics(self.state)
        event["resources"] = dict(self.state.resources)
        event["project_epoch"] = int(self.state.project_epoch)
        return event

    def run(
        self,
        horizon: int,
        *,
        probe_ready: Mapping[str, Any] | None = None,
        stop_on_exhaust: bool = True,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        # This call's cap: the ticks already spent plus its horizon. Passing the
        # horizon alone compared a per-call count with the project's cumulative
        # ticks, so from the second call on the budget check fired first and the
        # frontier and stagnation stops were never reached.
        cap = int(self.state.ticks) + max(0, int(horizon))
        for _ in range(max(0, int(horizon))):
            if self.state.resources.get('exhausted'):
                from .scheduler import admissible, local_execution
                local, _ = admissible(self.state, [a for a in self.eligible() if local_execution(a, self.state)])
                if not local:
                    events.append({'stage': 'terminate', 'reason': 'budget_exhausted',
                                   'research_status': 'BUDGET_EXHAUSTED', 'research_complete': False})
                    break
            if stop_on_exhaust:
                reason = should_terminate(self.state, horizon=cap)
                if reason and reason != "budget":
                    events.append({"stage": "terminate", "reason": reason})
                    break
            event = self.tick(probe_ready=probe_ready)
            events.append(event)
            result = event.get('result') or {}
            if result.get('status') == 'service_unavailable':
                events.append({'stage': 'terminate', 'reason': 'service_unavailable',
                               'research_status': 'SERVICE_UNAVAILABLE', 'research_complete': False,
                               'service': result.get('service', ''), 'detail': result.get('reason', ''),
                               'quota_exhausted': bool(result.get('quota_exhausted'))})
                break
        return events


def after_card(
    workspace: Path | None,
    card: GeneratedCard | Mapping[str, Any] | None,
    bundle: Mapping[str, Any] | None = None,
    *,
    world: Any = None,
) -> list[dict[str, Any]]:
    """Legacy mission hook. Sync state and emit one scheduler view.

    Does not decide generate → review → diagnose → experiment.
    Production research uses OpenWorld.run / run_research.
    """
    if workspace is None:
        return []
    env = OpenWorld.load(workspace)
    if card is not None:
        kind = idea_kind_of(card)
        claim = (
            str(card.get("claim") or "")
            if isinstance(card, Mapping)
            else str(getattr(card, "claim", "") or "")
        )
        qid = (
            str(card.get("card_id") or "")
            if isinstance(card, Mapping)
            else str(getattr(card, "card_id", "") or "")
        )
        lineage = (
            str(card.get("lineage_id") or "")
            if isinstance(card, Mapping)
            else str(getattr(card, "lineage_id", "") or "")
        )
        if kind == "question" and claim:
            env.state.add_question(
                question_id=qid or f"q-{len(env.state.open_questions)+1}",
                text=claim,
                required_capability=_card_required_capability(card),
                lineage_id=lineage,
                epistemic="GENERATED",
            )
        if kind == "acquire":
            env.state.note_missing(claim or "world")
        if world is not None:
            env.state.world_id = str(getattr(world, "id", "") or env.state.world_id)
    probe_ready = None
    if bundle and bundle.get("diagnosis"):
        diag = bundle["diagnosis"]
        lever = str(getattr(diag, "world_lever", "") or "")
        measure = str(getattr(diag, "world_observable", "") or "")
        if lever and measure:
            probe_ready = {
                "causal_handle": lever,
                "measure": measure,
                "freeze": env.state.world_id,
            }
    actions = env.eligible(probe_ready=probe_ready)
    chosen = env.schedule(actions)
    env.persist()
    return [
        {
            "stage": "openworld_tick",
            "eligible_types": [item.action_type for item in actions],
            "scheduled": None if chosen is None else chosen.to_dict(),
            "executed": False,
            "harness_version": env.harness.version_id,
            "world_id": env.state.world_id,
            "reason": "state_driven_affordance",
        }
    ]


def _card_required_capability(card: Any) -> str:
    fields = {}
    if isinstance(card, Mapping):
        raw = card.get("kind_fields")
        if isinstance(raw, Mapping):
            fields = dict(raw)
        fields.setdefault("required_world_state", card.get("required_world_state") or "")
        fields.setdefault("capability_gap", card.get("capability_gap") or "")
    else:
        raw = getattr(card, "kind_fields", None)
        if isinstance(raw, Mapping):
            fields = dict(raw)
    return str(
        fields.get("required_capability")
        or fields.get("capability_gap")
        or fields.get("required_world_state")
        or ""
    ).strip()


def dry_run_openworld(workspace: Path) -> list[dict[str, Any]]:
    """The required architecture chain, deterministic, no live API."""
    env = OpenWorld(workspace=workspace)
    env.state.goal = "does validator-version history change acceptance?"
    env.state.world_id = "W0"
    env.state.world_versions.append({"id": "W0", "world_id": "W0", "observables": ""})
    env.state.add_question(
        question_id="Q13",
        text="does validator-version history change acceptance?",
        required_capability="compare_validator_snapshots",
        lineage_id="line-q13",
        epistemic="GENERATED",
    )
    env.state.note_missing("compare_validator_snapshots")
    env.persist()
    events: list[dict[str, Any]] = []

    for _ in range(PATHOLOGY_FLOOR):
        observe = ActionInstance(
            SPECS[OBSERVE],
            target="Q13",
            reason="forced_observe_attempt",
            frontier_target_id="Q13",
            extra={
                "question": {
                    "id": "Q13",
                    "required_capability": "compare_validator_snapshots",
                    "text": env.state.goal,
                }
            },
        )
        events.append({"stage": "forced_observe", **env.execute(observe)})

    ev = env.tick()
    events.append(ev)
    if env.harness.version_id == "H0" and pathology_ready(env.pathologies):
        events.append({"stage": "forced_evolve", **env.evolve()})

    observe_now = ActionInstance(
        SPECS[OBSERVE],
        target="Q13",
        reason="capability_unlocked",
        frontier_target_id="Q13",
        extra={
            "question": {
                "id": "Q13",
                "required_capability": "compare_validator_snapshots",
                "text": env.state.goal,
            },
            "records": [
                {"validator_version": "v1", "ok": True},
                {"validator_version": "v2", "ok": False},
            ],
        },
    )
    events.append({"stage": "observe_after_h1", **env.execute(observe_now)})

    if env.state.open_questions or env.state.resolved_questions:
        theory = ActionInstance(
            SPECS[THEORIZE],
            target="Q13",
            reason="after_observation",
            lineage_id="line-q13",
            frontier_target_id="Q13",
        )
        events.append({"stage": "theory_eligible", **env.execute(theory)})
    env.persist()
    events.append(
        {
            "stage": "dry_run_summary",
            "harness_version": env.harness.version_id,
            "world_id": env.state.world_id,
            "capabilities": list(env.capabilities.names()),
            "resolved": [row.get("id") for row in env.state.resolved_questions],
            "metrics": dict(env.state.metrics),
        }
    )
    return events
