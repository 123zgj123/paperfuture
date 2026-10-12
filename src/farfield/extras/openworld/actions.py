"""Action / Affordance registry. Actions are choices; idea kinds are objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .state import ScientificState


SURVEY = "SURVEY"
ASK = "ASK"
OBSERVE = "OBSERVE"
ACQUIRE = "ACQUIRE"
THEORIZE = "THEORIZE"
PROBE = "PROBE"
EXPLORE = "EXPLORE"
VERIFY = "VERIFY"
VERIFY_REPLICATION = "VERIFY_REPLICATION"
VERIFY_MEASUREMENT = "VERIFY_MEASUREMENT"
VERIFY_ALTERNATIVE_EXPLANATION = "VERIFY_ALTERNATIVE_EXPLANATION"
VERIFY_WORLD_COMPATIBILITY = "VERIFY_WORLD_COMPATIBILITY"
VERIFY_NEGATIVE_RESULT = "VERIFY_NEGATIVE_RESULT"
VERIFY_HIGH_IMPACT_CLAIM = "VERIFY_HIGH_IMPACT_CLAIM"
REFLECT = "REFLECT"
SYNTHESIZE = "SYNTHESIZE"
ARCHIVE = "ARCHIVE"
FORK_LINEAGE = "FORK_LINEAGE"
EVOLVE_HARNESS = "EVOLVE_HARNESS"
# Project-level research control. SURVEY..SYNTHESIZE ask "what do we do with
# this question?". These three ask "is this project any good, and if not, what
# has to change?" — which is the decision a long-horizon agent actually faces.
REVIEW = "REVIEW"
REDESIGN = "REDESIGN"
PROJECT_PIVOT = "PROJECT_PIVOT"
# The Discovery Runtime. These produce, validate and evolve research ideas and
# the policy that chooses discovery moves. They never propose evidence, beliefs,
# questions or blocks: each may add only its own GENERATED artifact kind.
IDEATE = "IDEATE"
CRITIQUE = "CRITIQUE"
EVOLVE_POLICY = "EVOLVE_POLICY"

ACTION_TYPES = (
    SURVEY,
    ASK,
    OBSERVE,
    ACQUIRE,
    THEORIZE,
    PROBE,
    EXPLORE,
    VERIFY,
    REFLECT,
    SYNTHESIZE,
    ARCHIVE,
    FORK_LINEAGE,
    EVOLVE_HARNESS,
    REVIEW,
    REDESIGN,
    PROJECT_PIVOT,
    IDEATE,
    CRITIQUE,
    EVOLVE_POLICY,
)


@dataclass(frozen=True)
class ActionSpec:
    action_type: str
    preconditions: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    reads: tuple[str, ...]
    possible_outputs: tuple[str, ...]
    executor: str
    validator: str
    estimated_cost: float
    risk: str
    state_effect: str
    evidence_effect: str
    # What this action's executor may propose to the kernel: the event types,
    # and the kinds of artifact an `ArtifactAdded` from it may carry. The kernel
    # refuses anything else — a probe does not amend the contract or admit a
    # harness, and a reflection does not record a literature survey. Every
    # action may also report that it was blocked or failed (ALWAYS_ADMISSIBLE).
    emits: tuple[str, ...] = ()
    artifact_kinds: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "preconditions": list(self.preconditions),
            "required_capabilities": list(self.required_capabilities),
            "reads": list(self.reads),
            "possible_outputs": list(self.possible_outputs),
            "executor": self.executor,
            "validator": self.validator,
            "estimated_cost": self.estimated_cost,
            "risk": self.risk,
            "state_effect": self.state_effect,
            "evidence_effect": self.evidence_effect,
            "emits": list(self.emits),
            "artifact_kinds": list(self.artifact_kinds),
        }


# A report that an attempt was blocked or failed is admissible from any action;
# it is about the attempt, not about the world.
ALWAYS_ADMISSIBLE = ("ActionBlocked", "ActionFailed")


@dataclass(frozen=True)
class ActionInstance:
    spec: ActionSpec
    target: str = ""
    reason: str = ""
    lineage_id: str = ""
    frontier_target_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def action_type(self) -> str:
        return self.spec.action_type

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "target": self.target,
            "reason": self.reason,
            "lineage_id": self.lineage_id,
            "frontier_target_id": self.frontier_target_id,
            "extra": dict(self.extra),
        }


SPECS: dict[str, ActionSpec] = {
    SURVEY: ActionSpec(
        SURVEY,
        preconditions=("open_question_or_goal",),
        required_capabilities=(),
        reads=("scientific_archive", "literature"),
        possible_outputs=("citations", "gaps"),
        executor="survey_executor",
        artifact_kinds=("literature_survey",),
        validator="verify_papers",
        estimated_cost=0.4,
        risk="low",
        state_effect="record_citations",
        evidence_effect="none",
    ),
    ASK: ActionSpec(
        ASK,
        preconditions=("research_goal",),
        required_capabilities=(),
        reads=("working_memory",),
        possible_outputs=("question",),
        executor="ask_executor",
        emits=("QuestionCreated",),
        validator="question_schema",
        estimated_cost=0.2,
        risk="low",
        state_effect="add_open_question",
        evidence_effect="none",
    ),
    OBSERVE: ActionSpec(
        OBSERVE,
        preconditions=("open_question", "observable_present"),
        required_capabilities=(),
        reads=("world_version", "capability_registry"),
        possible_outputs=("QuestionEvidence",),
        executor="question_executor",
        emits=("EvidenceAdded", "QuestionResolved", "QuestionPartiallyResolved"),
        validator="observation_attestation",
        estimated_cost=0.5,
        risk="medium",
        state_effect="maybe_resolve_question",
        evidence_effect="question_evidence_if_attested",
    ),
    ACQUIRE: ActionSpec(
        ACQUIRE,
        preconditions=("capability_or_world_gap",),
        required_capabilities=(),
        reads=("missing_capabilities", "world_lineage"),
        possible_outputs=("AcquisitionEvidence", "world_version"),
        executor="acquire_executor",
        emits=("WorldCreated", "WorldAttested", "CapabilityProposed",
               "CapabilityValidated", "CapabilityTrusted", "EvidenceAdded"),
        artifact_kinds=("world_acquisition",),
        validator="world_attestation",
        estimated_cost=0.8,
        risk="medium",
        state_effect="new_world_version",
        evidence_effect="acquisition_evidence",
    ),
    THEORIZE: ActionSpec(
        THEORIZE,
        preconditions=("unexplained_phenomenon",),
        required_capabilities=(),
        reads=("open_questions", "evidence_records"),
        possible_outputs=("theory", "predictions"),
        executor="theory_discriminator",
        emits=("TheoryCreated", "EvidenceRejected"),
        validator="theory_schema",
        estimated_cost=0.3,
        risk="low",
        state_effect="add_generated_theory",
        evidence_effect="none",
    ),
    EXPLORE: ActionSpec(
        EXPLORE, preconditions=("exploratory_program",), required_capabilities=(),
        reads=("discovery_archive", "exploration_environment"),
        possible_outputs=("unvalidated_observation",), executor="exploration_executor",
        artifact_kinds=("exploration_execution", "native_evaluation", "native_comparison"), validator="execution_receipt_only",
        estimated_cost=0.0, risk="medium", state_effect="record_exploratory_execution",
        evidence_effect="none",
    ),
    PROBE: ActionSpec(
        PROBE,
        preconditions=("causal_handle", "valid_freeze", "measure"),
        required_capabilities=(),
        reads=("world_version", "harness_probe_executor"),
        possible_outputs=("ProbeEvidence",),
        executor="probe_executor",
        emits=("EvidenceAdded",),
        validator="fair_two_arm",
        estimated_cost=1.0,
        risk="high",
        state_effect="record_probe",
        evidence_effect="probe_evidence_if_world",
    ),
    VERIFY: ActionSpec(
        VERIFY,
        preconditions=("existing_evidence",),
        required_capabilities=(),
        reads=("evidence_id", "world_id"),
        possible_outputs=("confirmation_or_gap",),
        executor="verify_executor",
        emits=("EvidenceAdded", "EvidenceRejected"),
        validator="host_confirmation",
        estimated_cost=0.9,
        risk="medium",
        state_effect="confirmation_gaps",
        evidence_effect="same_evidence_id_only",
    ),
    REFLECT: ActionSpec(
        REFLECT,
        preconditions=("recent_events",),
        required_capabilities=(),
        reads=("failure_archive", "working_memory"),
        possible_outputs=("pathology_or_scientific_note",),
        executor="reflect_executor",
        artifact_kinds=("failure_lesson", "anomaly"),
        validator="classify_failure",
        estimated_cost=0.2,
        risk="low",
        state_effect="record_pathology_or_anomaly",
        evidence_effect="none",
    ),
    SYNTHESIZE: ActionSpec(
        SYNTHESIZE,
        preconditions=("evidence_or_theory",),
        required_capabilities=(),
        reads=("scientific_archive",),
        possible_outputs=("brief",),
        executor="synthesize_executor",
        emits=("FrontierChanged",),
        artifact_kinds=("research_synthesis",),
        validator="attested_writeup_lock",
        estimated_cost=0.6,
        risk="low",
        state_effect="draft_only",
        evidence_effect="none",
    ),
    ARCHIVE: ActionSpec(
        ARCHIVE,
        preconditions=("closed_or_negative_result",),
        required_capabilities=(),
        reads=("working_memory",),
        possible_outputs=("archive_row",),
        executor="archive_executor",
        artifact_kinds=("archive_record",),
        validator="epistemic_label",
        estimated_cost=0.1,
        risk="low",
        state_effect="move_to_archive",
        evidence_effect="none",
    ),
    FORK_LINEAGE: ActionSpec(
        FORK_LINEAGE,
        preconditions=("stagnation_or_diversity",),
        required_capabilities=(),
        reads=("scientific_state", "harness_version"),
        possible_outputs=("branches",),
        executor="fork_executor",
        emits=("LineageForked",),
        validator="generated_not_merged_as_world",
        estimated_cost=0.7,
        risk="medium",
        state_effect="open_branches",
        evidence_effect="verified_only_merge",
    ),
    EVOLVE_HARNESS: ActionSpec(
        EVOLVE_HARNESS,
        preconditions=("reproducible_harness_pathology",),
        required_capabilities=(),
        reads=("harness_archive", "failure_archive"),
        possible_outputs=("harness_version", "capability"),
        executor="harness_evolver",
        # A proposal only. Admission is a mission-boundary decision with its own
        # receipt; nothing proposed during research may install itself.
        emits=("HarnessPatchProposed",),
        validator="credit_gate",
        estimated_cost=0.9,
        risk="high",
        state_effect="harness_version_if_admitted",
        evidence_effect="none",
    ),
    REVIEW: ActionSpec(
        REVIEW,
        preconditions=("reviewable_claim_or_design",),
        required_capabilities=(),
        reads=("evidence_records", "quality_state", "research_contract"),
        possible_outputs=("review_records",),
        executor="review_executor",
        emits=("ReviewRecorded",),
        validator="structured_objection_schema",
        estimated_cost=0.3,
        risk="low",
        state_effect="record_review_debt",
        # A reviewer's agreement is not measurement. Review changes the plan; it
        # never changes what the world was observed to do.
        evidence_effect="none",
    ),
    REDESIGN: ActionSpec(
        REDESIGN,
        preconditions=("unresolved_objection_or_failed_design",),
        required_capabilities=(),
        reads=("reviews", "failed_designs", "quality_state"),
        possible_outputs=("design_revision",),
        executor="redesign_executor",
        emits=("DesignRevised",),
        validator="revision_addresses_root_cause",
        estimated_cost=0.4,
        risk="medium",
        state_effect="open_design_revision",
        evidence_effect="none",
    ),
    PROJECT_PIVOT: ActionSpec(
        PROJECT_PIVOT,
        preconditions=("direction_not_worth_continuing", "budget_headroom"),
        required_capabilities=(),
        reads=("research_contract", "quality_state", "resources", "reviews"),
        possible_outputs=("clean_restart_projection",),
        executor="pivot_executor",
        emits=("ProjectPivoted", "LineageRetired", "QuestionCreated"),
        artifact_kinds=("clean_restart_projection",),
        validator="preserves_world_evidence_and_contract",
        estimated_cost=0.5,
        risk="high",
        state_effect="retire_lineage_and_restart_from_contract",
        evidence_effect="none",
    ),
    # Discovery Runtime. No evidence effect, by construction: each may add only
    # its own GENERATED artifact kind, and the kernel refuses anything else.
    IDEATE: ActionSpec(
        IDEATE,
        preconditions=("research_intent",),
        required_capabilities=(),
        reads=("research_intent", "discovery_archive", "exploration_policy"),
        possible_outputs=("complete_idea",),
        executor="ideate_executor",
        artifact_kinds=("discovery_idea", "discovery_decision"),
        validator="complete_idea_schema",
        estimated_cost=0.3,
        risk="low",
        state_effect="add_generated_idea",
        evidence_effect="none",
    ),
    CRITIQUE: ActionSpec(
        CRITIQUE,
        preconditions=("unvalidated_idea",),
        required_capabilities=(),
        reads=("discovery_archive", "literature"),
        possible_outputs=("research_feedback_packet", "research_world_rollout"),
        executor="critique_executor",
        artifact_kinds=("idea_feedback",),
        validator="structural_validation_and_research_world_rollout",
        estimated_cost=0.3,
        risk="low",
        state_effect="record_generated_feedback",
        evidence_effect="none",
    ),
    EVOLVE_POLICY: ActionSpec(
        EVOLVE_POLICY,
        preconditions=("operator_outcomes_disagree_with_policy",),
        required_capabilities=(),
        reads=("discovery_archive", "exploration_policy"),
        possible_outputs=("exploration_policy_version",),
        executor="exploration_policy_evolver",
        artifact_kinds=("exploration_policy",),
        validator="policy_scope_and_archive_replay",
        estimated_cost=0.1,
        risk="low",
        state_effect="install_exploration_policy",
        evidence_effect="none",
    ),
}


def spec_of(action_type: str) -> ActionSpec:
    return SPECS[str(action_type)]


def _has_capability(names: Iterable[str], needed: str) -> bool:
    wanted = str(needed or "").strip()
    if not wanted:
        return True
    have = {str(item).strip() for item in names}
    return wanted in have


def _contrast_observations(state: Any, question: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Contrasts a live hypothesis named that the bound world can actually split.

    Read from the hypothesis's own `experiment_proposal`: the observable it
    predicts on, and the contrast lever the compiler bound it to. Both must be
    registered by the world, so this cannot invent a measurement — it only
    reaches one the world already offers.
    """
    qid = str(question.get("id") or "")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for theory in getattr(state, "theories", []) or []:
        if theory.get("role") == "competing_explanation":
            continue
        if str(theory.get("question_id") or "") != qid:
            continue
        proposal = dict(theory.get("experiment_proposal") or {})
        if not proposal:
            proposal = dict((dict(theory.get("card") or {}).get("research_metadata")
                             or {}).get("experiment_proposal") or {})
        outcomes = dict(proposal.get("expected_outcomes") or {})
        observables = [str(dict(row).get("observable") or "")
                       for row in outcomes.values() if isinstance(row, Mapping)]
        lever = str(dict(theory.get("card") or {}).get("lever")
                    or proposal.get("lever") or theory.get("lever") or "")
        for observable in observables:
            key = (lever, observable)
            if not lever or not observable or key in seen:
                continue
            seen.add(key)
            rows.append({"contrast": lever, "observable": observable,
                         "theory_id": str(theory.get("id") or "")})
    return rows


def _observable_present(state: ScientificState, question: Mapping[str, Any]) -> bool:
    required = str(question.get("required_observable") or "").strip()
    if required:
        blob = " ".join(
            str(row.get("observables") or row.get("id") or "")
            for row in state.world_versions
        )
        if required in blob or required in " ".join(state.available_capabilities):
            return True
        return False
    cap = str(question.get("required_capability") or "").strip()
    if cap:
        return _has_capability(state.available_capabilities, cap)
    return bool(state.world_id)


def _has_theory_for(state: ScientificState, question_id: str) -> bool:
    return any(
        str(row.get("question_id") or row.get("target") or "") == question_id
        or str(row.get("id") or "") == question_id
        for row in state.theories
    )


def sole_support(state: ScientificState, row: Mapping[str, Any]) -> bool:
    """Is this row the only admitted evidence its hypothesis rests on?

    The old test read ``high_impact`` and ``source_count`` from the evidence
    row. Nothing in the runtime wrote either field, so a whole class of
    verification debt — a consequential claim standing on one measurement —
    could never be raised. Whether a row stands alone is a fact about the state
    and is computed here rather than waited for.
    """
    tid = str(row.get("theory_id") or "")
    if not tid:
        return False
    supporting = {
        str(item.get("evidence_id") or "")
        for item in state.evidence_records
        if str(item.get("theory_id") or "") == tid
        and str(item.get("epistemic") or "") == "WORLD"
        and item.get("promotion_ok")
        and (item.get("outcome") or item.get("verdict")) == "supports"
    }
    return supporting == {str(row.get("evidence_id") or "")}


def replayable_by_default(row: Mapping[str, Any]) -> bool:
    """What the default verifier can recompute from the bound freeze alone.

    Dataset descriptives and an acquisition's digest are functions of the
    world's bytes. A registered experiment is not: replaying one needs the
    protocol runner a research worker provides.
    """
    return (str(row.get("measurement") or "") == "descriptives-v1"
            or str(row.get("role") or "") == "AcquisitionEvidence")


def verification_target(state: ScientificState) -> str:
    """The one evidence row a default VERIFY should check, named before it runs.

    A VERIFY used to be offered with no target and choose a row when it
    executed, so the row it checked could differ from the row whose debt had
    scheduled it — and a registered experiment it could not replay came back
    marked as a failed replay, which zeroes its reliability for good. The row
    is named here, where the frontier decides, and only rows this verifier can
    actually recompute are named. Contradictions and a claim standing on one
    experiment need an experiment rerun, which is a worker's VERIFY, not this.
    """
    for row in state.evidence_records:
        if (str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id")
                and replayable_by_default(row) and int(row.get("verify_count") or 0) < 1):
            return str(row["evidence_id"])
    return ""


def _needs_verify(state: ScientificState) -> bool:
    return bool(verification_target(state))


def _verify_kind(state: ScientificState) -> str:
    if state.contradictions:
        return VERIFY_ALTERNATIVE_EXPLANATION
    for row in state.evidence_records:
        if sole_support(state, row) and not row.get("replication_of"):
            return VERIFY_HIGH_IMPACT_CLAIM
        if str(row.get("outcome") or row.get("verdict") or "") in {"weakens", "negative"}:
            return VERIFY_NEGATIVE_RESULT
    return VERIFY_REPLICATION


def _causal_ready(state: ScientificState, extra: Mapping[str, Any] | None = None) -> bool:
    row = dict(extra or {})
    # A handle, a measure and a freeze. There was a second path here reading
    # ``causal_identification_ready``, which nothing ever set.
    return bool(row.get("causal_handle") and row.get("measure")
                and (row.get("freeze") or state.world_id))


def eligible_actions(
    state: ScientificState,
    *,
    capabilities: Iterable[str] = (),
    pathology_ready: bool = False,
    stagnation: bool = False,
    probe_ready: Mapping[str, Any] | None = None,
    workers_connected: bool = False,
    strategic_stagnation: Any = None,
    discovery: bool = False,
) -> list[ActionInstance]:
    """Deterministic eligibility. Not a pipeline stage list.

    ``discovery`` adds the discovery runtime's moves (IDEATE / CRITIQUE /
    EVOLVE_POLICY). They depend on neither literature nor a world, so no
    evidence precondition can withhold them.
    """
    caps = tuple(str(item) for item in capabilities) + tuple(state.available_capabilities)
    found: list[ActionInstance] = []
    goal_id = "goal:G0" if state.goal else ""

    if (state.goal or state.open_questions) and (
        state.ticks == 0 or not state.archive_ids
    ):
        found.append(
            ActionInstance(
                SPECS[SURVEY],
                target=state.goal,
                reason="goal_or_question",
                frontier_target_id=goal_id,
            )
        )
    if state.goal and not state.open_questions and not state.resolved_questions and not state.blocked_questions:
        found.append(
            ActionInstance(
                SPECS[ASK],
                target=state.goal,
                reason="research_goal",
                frontier_target_id=goal_id,
            )
        )

    for question in list(state.open_questions) + list(state.blocked_questions):
        qid = str(question.get("id") or "")
        lineage = str(question.get("lineage_id") or "")
        required_cap = str(question.get("required_capability") or "").strip()
        if _observable_present(state, question) and (
            not required_cap or _has_capability(caps, required_cap)
        ):
            found.append(
                ActionInstance(
                    SPECS[OBSERVE],
                    target=qid,
                    reason="observable_present",
                    lineage_id=lineage,
                    frontier_target_id=qid,
                    extra={"question": dict(question)},
                )
            )
            # A hypothesis may name an observational contrast the bound world
            # already registers. Without this the contrast is unreachable: the
            # generic OBSERVE returns dataset descriptives, and PROBE refuses a
            # question card because a stratum comparison is not an intervention.
            # A live run produced two well-formed contrasts and could execute
            # neither, so the whole observational family of questions — most of
            # empirical software engineering — dead-ended.
            for row in _contrast_observations(state, question):
                found.append(
                    ActionInstance(
                        SPECS[OBSERVE],
                        target=qid,
                        reason="named_contrast_on_the_bound_world",
                        lineage_id=lineage,
                        frontier_target_id=qid,
                        extra={"question": dict(question), **row},
                    )
                )
        else:
            gap = required_cap or str(question.get("required_observable") or "observable")
            found.append(
                ActionInstance(
                    SPECS[ACQUIRE],
                    target=qid,
                    reason="blocked_on_capability_or_world",
                    lineage_id=lineage,
                    frontier_target_id=qid,
                    extra={"gap": gap, "question": dict(question)},
                )
            )
        if not _has_theory_for(state, qid):
            found.append(
                ActionInstance(
                    SPECS[THEORIZE],
                    target=qid,
                    reason="unexplained_question",
                    lineage_id=lineage,
                    frontier_target_id=qid,
                )
            )

    for question in state.resolved_questions:
        qid = str(question.get("id") or "")
        if qid and not _has_theory_for(state, qid):
            found.append(
                ActionInstance(
                    SPECS[THEORIZE],
                    target=qid,
                    reason="discriminate_resolved_phenomenon",
                    lineage_id=str(question.get("lineage_id") or ""),
                    frontier_target_id=qid,
                )
            )

    acquireable = [name for name in state.missing_capabilities
                   if not str(name).startswith("executor:")]
    if acquireable and not any(item.action_type == ACQUIRE for item in found):
        found.append(
            ActionInstance(
                SPECS[ACQUIRE],
                target=acquireable[0],
                reason="capability_or_world_gap",
                frontier_target_id=state.open_questions[0]["id"] if state.open_questions else goal_id,
                extra={"gap": acquireable[0]},
            )
        )

    if (state.open_questions or state.anomalies) and not any(
        item.action_type == THEORIZE for item in found
    ):
        found.append(
            ActionInstance(
                SPECS[THEORIZE],
                reason="unexplained_phenomenon",
                frontier_target_id=goal_id,
            )
        )

    if _causal_ready(state, probe_ready):
        found.append(
            ActionInstance(
                SPECS[PROBE],
                target=str((probe_ready or {}).get("hypothesis") or ""),
                reason="causal_handle_and_freeze",
                frontier_target_id=str((probe_ready or {}).get("question_id") or goal_id),
                extra=dict(probe_ready or {}),
            )
        )

    if _needs_verify(state):
        found.append(
            ActionInstance(
                SPECS[VERIFY],
                target=verification_target(state),
                reason="existing_evidence",
                frontier_target_id=goal_id or (
                    str(state.evidence_records[-1].get("question_id") or "")
                    if state.evidence_records
                    else ""
                ),
                extra={"verify_kind": _verify_kind(state)},
            )
        )

    if state.ticks > 0:
        found.append(ActionInstance(SPECS[REFLECT], reason="recent_events"))

    last_synth = int((state.metrics or {}).get("last_synthesis_events") or 0)
    meaningful = int((state.metrics or {}).get("meaningful_events") or 0)
    if (state.evidence_records or state.theories) and (
        meaningful - last_synth >= 5 or state.contradictions or state.ticks == 1
    ):
        found.append(ActionInstance(SPECS[SYNTHESIZE], reason="evidence_or_theory"))

    if state.resolved_questions or state.failed_designs:
        found.append(ActionInstance(SPECS[ARCHIVE], reason="closed_or_negative_result"))

    if stagnation and len(state.active_lineages) < 4:
        from .executors import fork_candidates

        # A fork with nothing distinct to branch on is a wasted tick that then
        # blocks the next real fork. Only offer it when lanes exist.
        candidates = fork_candidates(state)
        # Diversifying is only worth a tick when the lanes are not already on
        # the menu. If every lane's next action is directly available, forking
        # renames work the controller can simply do.
        offered = {item.action_type for item in found}
        if len(candidates) >= 2 and not all(
                str(row.get("next_action") or "") in offered for row in candidates):
            found.append(
                ActionInstance(
                    SPECS[FORK_LINEAGE],
                    reason="stagnation_or_diversity",
                    extra={"fork_candidates": candidates},
                )
            )

    if pathology_ready:
        found.append(
            ActionInstance(
                SPECS[EVOLVE_HARNESS],
                reason="reproducible_harness_pathology",
            )
        )

    found.extend(project_actions(state, strategic_stagnation=strategic_stagnation))

    if workers_connected:
        found = worker_actions(state, found)
    if discovery:
        found.extend(discovery_actions(state))
        found.extend(requirement_actions(state))
        found.extend(judgment_actions(state))
        found.extend(reason_actions(state, found))
    # An action that provably cannot inform anything is not eligible. Spending a
    # tick to leave the state exactly as it was is a resource decision, and the
    # honest place to make it is here, before the action is ever offered.
    #
    # The stall bound belongs here for the same reason. It was originally applied
    # inside `unattempted_actions`, which `discovery_actions` never calls and
    # this function never called either — so the only thing consulting it was
    # `should_terminate`, and a live mission spent thirty calls on eight IDEATEs
    # and ten CRITIQUEs while the rule that existed to stop exactly that was
    # never asked. A bound that is not on the menu is not a bound.
    from .recovery import stalled

    # Stage shares belong after the scheduler's admission checks. A candidate
    # that will be refused cannot justify hiding the last runnable action.
    return [
        item for item in found
        if not information_precondition(state, item) and not stalled(state, item)]


def _open_reviews(state: ScientificState, severity: str = "") -> list[dict[str, Any]]:
    from .quality import open_reviews

    return open_reviews(state.reviews, severity=severity)


def _reviewable(state: ScientificState) -> bool:
    """Would a review actually say something that is not already recorded?

    Asked the weaker question — "is there anything reviewable?" — this returned
    true as soon as a hypothesis and a survey existed, and a live run spent a
    tick on a review that had no evidence to object to and returned
    ``no_new_objections``. The reviewer's own output is the only honest
    precondition, so it is computed and checked for something new.
    """
    from .executors import _structural_objections
    from .quality import result_review_scope

    scope = result_review_scope(state)
    if scope['evidence_ids'] and not any(r.get('result_review_scope') == scope['digest'] for r in state.reviews):
        return True

    known = {str(row.get("review_id") or "") for row in state.reviews}
    return any(str(row.get("review_id") or "") not in known
               for row in _structural_objections(state))


def _redesign_demands(state: ScientificState) -> list[dict[str, Any]]:
    """Objections and quality gaps that a rerun of the same design cannot fix.

    A redesign needs a design. After a pivot the previous epoch's evidence still
    leaves quality dimensions short — correctly, that evidence is preserved —
    but there is nothing in the new epoch to revise, and the honest next step is
    to build a design, not to revise one that was retired.
    """
    from .quality import REDESIGN_REPAIRS

    epoch = int(state.project_epoch)
    live_designs = [row for row in state.theories
                    if row.get("status") not in {"retired", "refuted"}
                    and row.get("role") != "competing_explanation"]
    demands: list[dict[str, Any]] = []
    addressed = {
        str(rid)
        for revision in state.design_revisions
        if int(revision.get("project_epoch") or 0) == epoch
        for rid in revision.get("addresses_reviews") or []
    }
    for review in _open_reviews(state):
        if review.get('theory_id') and not any(t.get('id') == review['theory_id'] for t in live_designs):
            continue
        repair = str(review.get("repair_type") or "")
        if repair not in REDESIGN_REPAIRS and repair != "stronger_baseline":
            continue
        if int(review.get("project_epoch") or 0) != epoch:
            continue  # an objection against a retired direction is history
        if str(review.get("review_id") or "") in addressed:
            continue
        demands.append({
            "kind": "review", "review_id": review.get("review_id"),
            "repair_type": repair, "dimension": review.get("dimension"),
            "objection": review.get("objection"), "theory_id": review.get("theory_id"),
            "question_id": review.get("question_id"), "severity": review.get("severity"),
        })
    dimensions = (state.quality or {}).get("dimensions") or {}
    for name, row in sorted(dimensions.items()):
        if not live_designs:
            break  # nothing to revise; the frontier owns this gap
        if "REDESIGN" not in (row.get("required_next_work") or []):
            continue
        if any(item.get("dimension") == name for item in demands):
            continue
        if any(str(revision.get("dimension") or "") == name
               and int(revision.get("project_epoch") or 0) == int(state.project_epoch)
               for revision in state.design_revisions):
            continue
        demands.append({
            "kind": "quality_gap", "dimension": name, "severity": "major",
            "repair_type": "experiment_redesign",
            "objection": "; ".join(row.get("blocking_reasons") or []) or f"{name}_below_contract",
        })
    return demands


def _pivot_trigger(state: ScientificState, *, strategic_stagnation: Any = None) -> dict[str, Any] | None:
    """When is the honest move to stop repairing and change direction?

    Not "the experiment failed" — that is REDESIGN. This is the case where even
    a successful next experiment would not make the project worth finishing:
    the direction carries an unrepairable objection, its redesigns are spent, or
    the whole hypothesis set has drifted off the contract's root question.
    """
    from .quality import PIVOT_REPAIRS
    from .resources import budget_phase

    epoch = int(state.project_epoch)
    # A direction cannot be abandoned before it has been tried. After a pivot the
    # fresh epoch has to do real work before it may be abandoned in turn,
    # otherwise the controller can pivot forever without researching anything.
    epoch_work = [row for row in state.artifacts
                  if row.get("kind") == "action_attempt"
                  and str(row.get("action_type") or "") in {"PROBE", "VERIFY", "OBSERVE"}
                  and int(row.get("project_epoch") or 0) == epoch]
    if epoch and not epoch_work:
        return None

    demanded = [r for r in _open_reviews(state)
                if str(r.get("repair_type") or "") in PIVOT_REPAIRS
                and int(r.get("project_epoch") or 0) == epoch]
    if demanded:
        return {"reason": "reviewer_demanded_project_pivot",
                "review_ids": [r.get("review_id") for r in demanded]}

    # Opening a new direction costs a full research arc. With the budget nearly
    # gone that is not courage, it is guaranteeing two unfinished projects.
    phase = budget_phase(state)
    if phase in {"closing", "exhausted"}:
        return None

    revisions = [r for r in state.design_revisions if int(r.get("project_epoch") or 0) == epoch]
    fatal = [r for r in _open_reviews(state, severity="fatal")
             if int(r.get("project_epoch") or 0) == epoch]
    quality = state.quality or {}
    central = ((quality.get("dimensions") or {}).get("central_claim_support") or {})
    executed_revisions = [r for r in revisions if r.get("experiment_attempted_by")]
    exhausted = len(executed_revisions) >= 2 and all(
        str(r.get("outcome") or "open") != "resolved" for r in executed_revisions)

    if fatal and exhausted:
        return {"reason": "fatal_objection_survived_the_available_redesigns",
                "review_ids": [r.get("review_id") for r in fatal],
                "revisions_tried": len(executed_revisions)}
    if "PROJECT_PIVOT" in (central.get("required_next_work") or []):
        return {"reason": "no_active_hypothesis_still_serves_the_root_question",
                "blocking_reasons": list(central.get("blocking_reasons") or [])}
    if strategic_stagnation is not None and exhausted:
        reasons = list(getattr(strategic_stagnation, "reasons", ()) or [])
        return {"reason": "strategic_stagnation_after_exhausted_redesign", "detail": reasons}
    return None


def _action_variant(action: ActionInstance) -> str:
    """What distinguishes this action from a sibling of the same type and target.

    Three THEORIZE actions on one question differ only by research lane. Keying
    established outcomes on type and target alone let one lane's failure ban the
    two lanes that were never tried.
    """
    import json

    row = {name: action.extra.get(name) for name in
           ("research_lane", "hop", "parent_theory_id", "theory_id",
            "replication_world_id", "verify_kind")}
    # Discovery moves differ by mode, operator and lineage. Only they carry
    # these keys, so every other action's variant string is unchanged.
    if action.action_type in {IDEATE, CRITIQUE, EVOLVE_POLICY}:
        row.update({name: action.extra[name] for name in
                    ("mode", "operator", "parent_idea_id", "idea_id", "lineage_id", "branch_kind")
                    if name in action.extra})
    # A judgment panel is about one subject; a panel on another is another action.
    if action.action_type == REVIEW and action.extra.get("judgment_panel"):
        row["judgment_subject"] = action.extra.get("subject_digest")
    return json.dumps(row, sort_keys=True, default=str)


def information_precondition(state: ScientificState, action: ActionInstance) -> str:
    """Could this action tell the project anything it does not already know?

    A value precondition, not a score. Some actions always "succeed" and
    routinely change nothing — REFLECT with no unexamined failure, ARCHIVE with
    nothing newly closed, REVIEW over evidence every objection already covers.
    Offering them costs a tick and a call and returns the state unchanged, so
    they are refused at the eligibility layer where preconditions belong.

    Returns a refusal reason, or "" when the action can still inform something.
    """
    kind = action.action_type
    session_start = max((i for i, row in enumerate(state.artifacts)
                         if row.get('kind') == 'research_service_session'), default=-1)
    attempts = [row for i, row in enumerate(state.artifacts) if row.get("kind") == "action_attempt"
                and (row.get('status') not in {'service_unavailable', 'budget_exhausted'} or i >= session_start)]

    # The general rule, before any per-action one. If this exact action on this
    # exact target has already come back twice with the same outcome and
    # admitted nothing, the project has learned that fact. Running it a third
    # time is not an experiment, it is a re-read of a known answer — and this
    # holds even when unrelated parts of the state have moved on.
    variant = _action_variant(action)
    inert = [row for row in attempts
             if str(row.get("action_type") or "") == kind
             and str(row.get("target") or "") == str(action.target or "")
             and str(row.get("variant") or "") == variant
             and int(row.get("project_epoch") or 0) == int(state.project_epoch)
             and not int(row.get("events_admitted") or 0)]
    outcomes = [str(row.get("outcome_key") or "") for row in inert]
    for outcome in set(outcomes):
        if outcome and outcomes.count(outcome) >= 2:
            return f"already_established_that_this_returns_{outcome}"

    if kind == REFLECT:
        # Reflection needs an unexamined failure to reflect on.
        if not state.failed_designs and not state.anomalies:
            return "nothing_has_failed_yet"
        known = {str(row.get("signature") or "") for row in state.artifacts
                 if row.get("kind") == "failure_lesson"}
        signatures = {"|".join(str(row.get(key) or "") for key in
                               ("action_type", "theory_id", "reason"))
                      for row in state.failed_designs}
        if signatures and signatures <= known:
            return "every_recorded_failure_already_has_a_lesson"

    if kind == ARCHIVE:
        closed = sorted(str(row.get("id") or "") for row in state.resolved_questions)
        failed = sorted({"|".join(str(row.get(key) or "") for key in ("action_type", "theory_id"))
                         for row in state.failed_designs})
        if not (closed or failed):
            return "nothing_is_closed_to_archive"
        previous = next((row for row in reversed(state.artifacts)
                         if row.get("kind") == "archive_record"), {})
        if (list(previous.get("closed_questions") or []) == closed
                and list(previous.get("closed_designs") or []) == failed):
            return "the_archive_already_records_this_set"

    if kind == REVIEW and action.extra.get('judgment_panel'):
        from .judgment import judgment_due
        if not judgment_due(state):
            return "this_subject_has_already_been_judged"
    elif kind == REVIEW and not action.extra.get('program_review') and not _reviewable(state):
        return "the_reviewer_has_nothing_to_add_that_is_not_already_recorded"

    if kind == SYNTHESIZE:
        gap = list((state.quality or {}).get("quality_gap") or [])
        drafted = {str(row.get("evidence_id") or "") for row in state.artifacts
                   if row.get("kind") == "research_synthesis"}
        measured = {str(row.get("evidence_id") or "") for row in state.evidence_records
                    if str(row.get("role") or "") == "ProbeEvidence"}
        if gap and measured and measured <= drafted:
            return "a_redraft_does_not_close_an_open_evidence_gap"

    return ""


def project_actions(
    state: ScientificState, *, strategic_stagnation: Any = None
) -> list[ActionInstance]:
    """REVIEW / REDESIGN / PROJECT_PIVOT eligibility from canonical state."""
    found: list[ActionInstance] = []
    if _reviewable(state):
        found.append(
            ActionInstance(
                SPECS[REVIEW],
                target=str(state.goal or ""),
                reason="reviewable_claim_or_design",
                frontier_target_id="goal:G0" if state.goal else "",
                extra={"project_epoch": int(state.project_epoch)},
            )
        )
    for demand in _redesign_demands(state):
        found.append(
            ActionInstance(
                SPECS[REDESIGN],
                target=str(demand.get("theory_id") or demand.get("dimension") or ""),
                reason=f"unresolved_{demand.get('kind')}",
                frontier_target_id=str(demand.get("question_id") or "goal:G0"),
                extra={"demand": demand, "project_epoch": int(state.project_epoch)},
            )
        )
    trigger = _pivot_trigger(state, strategic_stagnation=strategic_stagnation)
    if trigger is not None:
        found.append(
            ActionInstance(
                SPECS[PROJECT_PIVOT],
                target=str(state.goal or ""),
                reason=str(trigger.get("reason") or "direction_not_worth_continuing"),
                frontier_target_id="goal:G0" if state.goal else "",
                extra={"trigger": trigger, "project_epoch": int(state.project_epoch)},
            )
        )
    return found


def rejected_execution(state: Any, execution: Mapping[str, Any], world_digest: str,
                       study_stage: str) -> bool:
    """A renamed claim cannot rehabilitate a rejected measurement identity.

    Invalidation remains historical evidence. New interventions, measurement
    definitions, or frozen world bytes can be tested; identical rejected work
    needs a new, verifiable repair identity instead of another EvidenceAdded.
    """
    keys = ('operation', 'parameters', 'metric', 'expected_direction', 'margin',
            'min_sample_size', 'study')
    definition = {key: execution.get(key) for key in keys}
    rows = state.get('evidence_records', []) if isinstance(state, Mapping) else state.evidence_records
    for row in rows:
        if (not row.get('invalidated') or row.get('epistemic') != 'WORLD'
                or row.get('world_digest') != world_digest or row.get('study_stage') != study_stage):
            continue
        registered = (row.get('registration') or {}).get('execution')
        if isinstance(registered, Mapping) and definition == {key: registered.get(key) for key in keys}:
            return True
    return False


def requirement_actions(state: ScientificState) -> list[ActionInstance]:
    """Claim-derived experiment contracts become acquire / pilot / confirmatory.

    Worlds do not write the question. A missing world is first an acquire/build
    problem. Pilot is a PROBE stage, not a new action type.
    """
    from .discovery import (
        EMPIRICAL_CLAIMS,
        NON_DISCRIMINATING,
        discovery_view,
        experiment_discriminates,
        failure_explanations_for,
        freeze_branch_commit,
        has_failure_diagnosis,
        idea_of,
    )
    from .requirement import (
        CALIBRATION,
        CAN_ACQUIRE_WORLD,
        CAN_BUILD_WORLD,
        CAPABILITY_GAP,
        CHEAP_STAGES,
        CONFIRMATORY,
        WORLD_EXTENSION,
        MATCHED_EXISTING_WORLD,
        PILOT,
        PILOT_INCONCLUSIVE,
        PILOT_NEGATIVE,
        PILOT_POSITIVE,
        REPLICATION,
        RESOURCE_BLOCKED,
        SCIENTIFIC_CHEAP_STAGES,
        SCIENTIFIC_PILOT,
        UNSUPPORTED_WORLD,
        acquired_world_id,
        cheap_execution_stage,
        classify_world_match,
        protocol_runnable,
        research_promise,
        requirement_of_idea,
    )

    view = discovery_view(state)
    if not view["ideas"]:
        return []
    found: list[ActionInstance] = []
    goal_id = "goal:G0" if state.goal else ""
    default_qid = str((state.open_questions[0] or {}).get("id") or goal_id) if state.open_questions else goal_id
    from ..validity import evidence_eligible, corroborates
    eligible = [row for row in state.evidence_records if evidence_eligible(row)]
    cheap = [row for row in eligible
             if row.get("study_stage") in CHEAP_STAGES]
    confirmations = [row for row in eligible
                     if row.get("study_stage") == CONFIRMATORY]
    replications = [row for row in eligible
                    if row.get("study_stage") == REPLICATION]
    for row in view["ideas"]:
        idea_question = str(row.get('question_id') or default_qid)
        current_question = next((q for q in state.open_questions + state.partial_questions
                                 + state.blocked_questions + state.resolved_questions
                                 if q.get('id') == idea_question), {})
        if current_question.get('transition_status') == 'reconsideration_required':
            continue
        qid = idea_question
        fb = view["feedback"].get(str(row["id"])) or {}
        if idea_of(row).claim_type not in EMPIRICAL_CLAIMS:
            continue
        req = requirement_of_idea(state, row)
        bound_id = acquired_world_id(state, req)
        bound = None
        if bound_id:
            from pathlib import Path
            from ..world import load_fixture
            version = next((w for w in state.world_versions if (w.get('id') or w.get('world_id')) == bound_id), {})
            try:
                bound = load_fixture(Path(version['path']))
            except (KeyError, OSError, ValueError):
                pass
        match = classify_world_match(req, bound=bound) if bound else classify_world_match(req)
        runnable = protocol_runnable(req, match)
        promise = research_promise(fb)
        from .discovery import internal_operational_validation
        internal_validation = internal_operational_validation(state, row, fb)
        scientific_run = promise['promising'] or internal_validation
        extra = {
            "discovery_idea_id": str(row["id"]),
            "experiment_requirement": req.to_dict(),
            "world_match": match,
            "protocol_runnable": runnable,
            "research_promise": promise,
        }
        if match['status'] in {RESOURCE_BLOCKED, UNSUPPORTED_WORLD}:
            missing = 'experiment_feasibility:' + str(match['status']).lower()
            if missing not in state.missing_capabilities:
                found.append(ActionInstance(SPECS[ACQUIRE], target=qid, frontier_target_id=qid,
                    reason='registered_experiment_requires_unavailable_resources_or_world',
                    extra={**extra, 'gap': 'capability_gap:' + missing,
                           'capability_name': missing, 'gap_kind': 'feasibility'}))
            continue
        if req.kind == WORLD_EXTENSION:
            action_name = str(req.extension.get("action") or "extension")
            if bound_id:
                continue
            if req.extension.get("implementation") and action_name not in state.available_capabilities:
                if action_name not in state.missing_capabilities:
                    found.append(ActionInstance(
                        SPECS[ACQUIRE], target=qid, frontier_target_id=qid,
                        reason="world_extension_awaits_admission",
                        extra={**extra, "gap": f"world_extension:{action_name}",
                               "capability_name": action_name}))
            continue
        if bound_id and match["status"] == CAPABILITY_GAP:
            match = {
                **match,
                "status": MATCHED_EXISTING_WORLD,
                "world_id": bound_id,
                "reason": "constructed_or_attested_world_already_bound",
            }
            runnable = protocol_runnable(req, match)
            extra = {**extra, "world_match": match, "protocol_runnable": runnable}
        if match["status"] == CAPABILITY_GAP:
            missing = dict(match.get("missing") or {})
            needed = [name for name, flag in missing.items() if flag
                      and name in {"missing_intervention", "missing_observable", "missing_evaluator"}]
            # A proposal that wrote no protocol is missing every one of these by
            # construction. That says nothing about the world: a live proposal
            # described connectivity survival in words, left `experiment` empty,
            # and the project went to ACQUIRE `executor:simulation` — an executor
            # it already has. The acquisition could not attest itself, the only
            # question was marked `required_instrument_unavailable`, and the run
            # ended in three ticks beside the instrument it said was missing.
            # An un-operationalised proposal is discovery's to repair, with the
            # instrument list in its brief; it is not a capability to acquire.
            from .experiment import OPERATIONS
            unwritten = not dict(req.execution or {}) and not dict(req.extension or {})
            class_has_instrument = any(
                klass == req.world_class for klass, _ in OPERATIONS.values())
            if needed and unwritten and class_has_instrument:
                continue
            if needed:
                cap_name = f"executor:{req.world_class or 'unclassified'}"
                if cap_name not in state.missing_capabilities and cap_name not in state.available_capabilities:
                    found.append(ActionInstance(
                        SPECS[ACQUIRE], target=qid, frontier_target_id=qid,
                        reason="missing_measurement_for_discrimination",
                        extra={**extra, "gap": f"capability_gap:{cap_name}",
                               "capability_name": cap_name,
                               "needed_for_discrimination": needed}))
            continue
        if not runnable["runnable"]:
            # The gap is recorded by `frontier.measurement_gaps` and carried to
            # the generator by the research journal. The repair itself belongs to
            # whoever can rewrite the protocol — a REDESIGN produces a design
            # revision and leaves the frozen experiment exactly as it was, which
            # a live run demonstrated by redesigning twice and changing nothing.
            continue
        idea_id = str(row["id"])
        if bound_id and match['status'] not in {RESOURCE_BLOCKED, UNSUPPORTED_WORLD}:
            match = {
                **match,
                "status": MATCHED_EXISTING_WORLD,
                "world_id": bound_id,
                "reason": "constructed_or_attested_world_already_bound",
            }
        extra = {
            "discovery_idea_id": idea_id,
            "experiment_requirement": req.to_dict(),
            "world_match": match,
            "protocol_runnable": runnable,
            "research_promise": promise,
        }
        stage = SCIENTIFIC_PILOT if internal_validation else cheap_execution_stage(fb)
        idea_cheap = [item for item in cheap if item.get("discovery_idea_id") == idea_id]
        idea_pilots = [item for item in idea_cheap if item.get("study_stage") in SCIENTIFIC_CHEAP_STAGES]
        idea_calibrations = [item for item in idea_cheap if item.get("study_stage") == CALIBRATION]
        tracked = idea_pilots if scientific_run else idea_calibrations
        idea_confirm = [item for item in confirmations if item.get("discovery_idea_id") == idea_id]
        if match["status"] in {CAN_BUILD_WORLD, CAN_ACQUIRE_WORLD} and not tracked and not bound_id:
            gap = (f"construct_template:{req.world_class}" if match["status"] == CAN_BUILD_WORLD
                   else (match.get("world_id") or req.world_class))
            if match["status"] == CAN_ACQUIRE_WORLD and not evidence_world_for(state, match.get("world_id")):
                gap = f"construct_template:{req.world_class}"
            found.append(ActionInstance(
                SPECS[ACQUIRE], target=qid, frontier_target_id=qid,
                reason="requirement_needs_executable_world",
                extra={**extra, "gap": gap}))
            continue
        freeze = bound_id or state.world_id
        offered = PILOT if stage == SCIENTIFIC_PILOT else CALIBRATION
        from .study import setting_ids, study_of
        study_settings = setting_ids(req.execution) if study_of(req.execution) is not None else []
        if not tracked:
            found.append(ActionInstance(
                SPECS[PROBE], target=idea_id, frontier_target_id=qid,
                reason=("cheap_external_pilot" if scientific_run
                        else "calibration_of_runnable_fixture"),
                extra={**extra, "study_stage": offered, "execution_kind": stage,
                       "causal_handle": req.primary_metric,
                       "measure": req.primary_metric, "freeze": freeze,
                       # A study pilots one registered setting. The pilot shows
                       # the protocol runs; it never stands in for the rest.
                       "study_setting_id": study_settings[0] if study_settings else ""}))
            continue
        negatives = [item for item in tracked if item.get("pilot_verdict") == PILOT_NEGATIVE]
        inconclusives = [item for item in tracked if item.get("pilot_verdict") == PILOT_INCONCLUSIVE]
        insufficient = [item for item in tracked if any(c.get('check') == 'sample'
            and c.get('status') in {'fail', 'undefined'} for c in (item.get('validity') or {}).get('checks', []))]
        if insufficient:
            if not has_failure_diagnosis(state, idea_id):
                found.append(ActionInstance(SPECS[VERIFY], target=str(insufficient[-1]['evidence_id']),
                    frontier_target_id=qid, reason='diagnose_registered_sample_gap',
                    extra={**extra, 'diagnostic': True, 'verify_kind': 'VERIFY_MEASUREMENT'}))
            continue
        if negatives:
            if not has_failure_diagnosis(state, idea_id):
                target = str(negatives[-1].get("evidence_id") or idea_id)
                found.append(ActionInstance(
                    SPECS[VERIFY], target=target, frontier_target_id=qid,
                    reason="diagnose_competing_failure_explanations",
                    extra={**extra, "diagnostic": True,
                           "verify_kind": "VERIFY_MEASUREMENT",
                           "failure_explanations": failure_explanations_for(negatives),
                           "branch_priority": 2.4}))
                continue
        if inconclusives and not any(item.get("pilot_verdict") == PILOT_POSITIVE for item in tracked):
            if not has_failure_diagnosis(state, idea_id):
                found.append(ActionInstance(
                    SPECS[VERIFY], target=str(inconclusives[-1]['evidence_id']), frontier_target_id=qid,
                    reason='diagnose_insufficient_discrimination',
                    extra={**extra, 'diagnostic': True, 'verify_kind': 'VERIFY_MEASUREMENT'}))
            continue
        if not scientific_run:
            continue
        from .experiment import registration, digest
        registered = [item for item in idea_pilots
                      if item.get('registration_digest') == digest(registration(req))
                      and item.get('world_id') == freeze and item.get('epistemic') == 'WORLD'
                      and item.get("study_stage") in SCIENTIFIC_CHEAP_STAGES]
        directional = any(item.get("pilot_verdict") == PILOT_POSITIVE or (
            item.get('pilot_verdict') == PILOT_NEGATIVE and item.get('diagnosed') and item.get('replay_ok'))
            for item in registered)
        registered_confirm = [item for item in idea_confirm
                              if item.get('registration_digest') == digest(registration(req))]
        covered = {str(item.get('study_setting_id') or '') for item in registered_confirm}
        # Without a study there is one confirmation to run; with one there is
        # exactly one per registered setting, and the study is unfinished until
        # each has its own admitted reading.
        pending = ([sid for sid in study_settings if sid not in covered] if study_settings
                   else ([] if idea_confirm else [""]))
        if directional and pending:
            idea = idea_of(row)
            # An operational confirmation establishes only its registered
            # comparison. Mechanism discrimination needs structured forecasts.
            predictions = idea.predictions
            if ((predictions and not experiment_discriminates(predictions))
                    or (idea.competing_explanation and
                        idea.competing_explanation == idea.validation_direction)):
                extra = {**extra, "discrimination": NON_DISCRIMINATING}
                continue
            found.append(ActionInstance(
                SPECS[PROBE], target=idea_id, frontier_target_id=qid,
                reason=("registered_study_setting" if study_settings else
                        "confirmatory_after_positive_pilot" if not negatives
                        else "confirmatory_after_diagnosed_negative_pilot"),
                extra={**extra, "study_stage": CONFIRMATORY,
                       "causal_handle": req.primary_metric, "measure": req.primary_metric,
                       "freeze": freeze, "study_setting_id": pending[0],
                       "study_settings_remaining": len(pending),
                       "branch_commit": freeze_branch_commit({
                           "scientific_hypothesis": idea.central_claim,
                           "decisive_uncertainty": "which_prediction_the_bound_world_matches",
                           "budget": 8,
                       }, why=idea.central_claim)}))
            continue
        if study_settings and pending:
            # A study that has not covered every registered setting has nothing
            # to replicate yet.
            continue
        valid_confirm = [item for item in idea_confirm if item.get('stage_verdict') == 'POSITIVE'
                         and corroborates(item)
                         and item.get('registration_digest') == digest(registration(req))]
        if valid_confirm and not [item for item in replications if item.get("discovery_idea_id") == idea_id]:
            primary = valid_confirm[-1]
            for world in state.world_versions:
                if (world.get('research_role') != 'heldout' or not world.get('attested')
                        or world.get('digest') == primary.get('world_digest')
                        or world.get('schema') != req.schema or not world.get('path')):
                    continue
                from ..world import load_fixture
                from pathlib import Path
                try:
                    held = load_fixture(Path(world['path']))
                    if classify_world_match(req, bound=held)['status'] != MATCHED_EXISTING_WORLD:
                        continue
                    from .experiment import scientific_payload_digest
                    primary_path = next((w.get('path') for w in state.world_versions
                                         if w.get('id') == primary.get('world_id')), '')
                    original = load_fixture(Path(primary_path))
                    if scientific_payload_digest(original, req.execution['operation'], req.execution['parameters']) == scientific_payload_digest(held, req.execution['operation'], req.execution['parameters']):
                        continue
                except (OSError, ValueError, KeyError):
                    continue
                found.append(ActionInstance(
                    SPECS[PROBE], target=idea_id, frontier_target_id=qid,
                    reason="heldout_replication_after_confirmatory",
                    extra={**extra, "study_stage": REPLICATION,
                           "replication_world_id": held.id,
                           "replication_of": str(primary.get("evidence_id") or ""),
                           "causal_handle": req.primary_metric, "measure": req.primary_metric,
                           "freeze": held.id}))
                break
    active = []
    for action in found:
        world = next((w for w in state.world_versions
                      if (w.get('id') or w.get('world_id')) == action.extra.get('freeze')), {})
        execution = (action.extra.get('experiment_requirement') or {}).get('execution') or {}
        if action.action_type == PROBE and rejected_execution(
                state, execution, str(world.get('digest') or world.get('world_digest') or ''),
                str(action.extra.get('study_stage') or '')):
            continue
        active.append(action)
    return active


def _registered_experiment_pending(state: ScientificState, qid: str) -> bool:
    """Is a registered experiment on this question still waiting to be settled?

    Two ways of measuring exist for one question: the registered protocol —
    frozen, replayable, bound to a world digest — and a model-written probe of a
    hypothesis card. While the first is runnable and unfinished, the second is
    not offered. A live run calibrated a registered `connectivity_survival`
    experiment and then spent two paid probes on card experiments that came back
    `uninformative` with no registration at all, while the registered pilot
    waited. One measurement path at a time, and the registered one goes first.
    """
    from .discovery import discovery_view

    try:
        ideas = discovery_view(state)["ideas"]
    except (AttributeError, KeyError, TypeError):
        return False
    for row in ideas:
        if str(row.get("question_id") or qid) != str(qid):
            continue
        experiment = (row.get("idea") or {}).get("experiment")
        if not (isinstance(experiment, Mapping) and experiment.get("operation")):
            continue
        idea_id = str(row.get("id") or "")
        settled = any(
            str(item.get("discovery_idea_id") or "") == idea_id
            and str(item.get("study_stage") or "") in {"confirmatory", "replication"}
            and not item.get("invalidated")
            for item in state.evidence_records)
        if not settled:
            return True
    return False


def evidence_world_for(state: ScientificState, world_id: str | None) -> bool:
    wid = str(world_id or "")
    return bool(wid) and any(
        str(row.get("id") or row.get("world_id") or "") == wid
        and row.get("attested") is not False
        for row in state.world_versions
    )


def discovery_actions(state: ScientificState) -> list[ActionInstance]:
    """The discovery runtime's next moves, planned from its own records.

    Each targets the root goal, so no question's status can withhold it, and
    each carries in ``extra`` the mode, operator and lineage the plan chose.
    """
    from .discovery import plan_moves

    goal_id = "goal:G0" if state.goal else ""
    return [
        ActionInstance(SPECS[move["action"]], target=str(move["target"]),
                       reason=str(move["reason"]),
                       lineage_id=str(move["extra"].get("lineage_id") or ""),
                       frontier_target_id=goal_id, extra=dict(move["extra"]))
        for move in plan_moves(state)
    ]


def judgment_actions(state: ScientificState) -> list[ActionInstance]:
    """Convene the judgment panel when the contract asks for a grade nobody has given.

    Offered only beside the discovery runtime, because it needs a model, and only
    once a final reading exists, because before that there is nothing to judge
    but a proposal. It is a REVIEW: same action, same admission, same reducer.
    """
    from .judgment import judgment_due, judgment_subject

    due = judgment_due(state)
    if not due:
        return []
    return [ActionInstance(
        SPECS[REVIEW], target=str(state.goal or ""), reason="required_judgment_has_no_judge",
        frontier_target_id="goal:G0" if state.goal else "",
        extra={"judgment_panel": True, "dimensions": due,
               "subject_digest": str(judgment_subject(state).get("digest") or ""),
               "project_epoch": int(state.project_epoch)})]


def reason_actions(state: ScientificState, already: list[ActionInstance] | None = None) -> list[ActionInstance]:
    """Inject LOCAL_RESEARCH / PROGRAM_REVIEW moves. Same loop, different mode."""
    from .discovery import reason
    offered = {(item.action_type, item.reason, item.target) for item in (already or [])}
    found: list[ActionInstance] = []
    judgment = reason(state)
    goal_id = "goal:G0" if state.goal else ""
    for move in judgment.get("moves") or []:
        key = (str(move.get("action") or ""), str(move.get("reason") or ""),
               str(move.get("target") or ""))
        if key in offered:
            continue
        spec = SPECS.get(str(move.get("action") or ""))
        if spec is None:
            continue
        found.append(ActionInstance(
            spec, target=str(move.get("target") or ""),
            reason=str(move.get("reason") or ""),
            frontier_target_id=goal_id,
            extra=dict(move.get("extra") or {}),
        ))
    return found


def worker_actions(state: ScientificState, actions: list[ActionInstance]) -> list[ActionInstance]:
    """Readiness from persisted scientific artifacts, not a second pipeline."""
    from .discovery import bridge_candidates, evidence_world_bound

    # Registered hypotheses are compiled for a world an experiment can run on.
    # Without one, a first-hop lane only manufactured a hypothesis whose probe
    # was certain to block, and the block retired the question. Ideas for a
    # project with no world yet belong to the discovery runtime.
    world_bound = evidence_world_bound(state)
    surveys = [r for r in state.artifacts if r.get("kind") == "literature_survey" and r.get("works")]
    # The control-plane observer is still useful, but cannot substitute for survey.
    found = [a for a in actions if a.action_type not in {SURVEY, THEORIZE, PROBE, VERIFY, SYNTHESIZE}]
    found = [a for a in found if a.action_type != OBSERVE or not any(
        r.get("role") == "QuestionEvidence" and r.get("question_id") == a.target
        and r.get("world_id") == state.world_id for r in state.evidence_records)]
    for row in state.evidence_records:
        if row.get("study_stage") in {"pilot", "calibration", "scientific_pilot"}:
            continue
        if row.get("epistemic") == "WORLD" and not row.get("verify_count"):
            found.append(ActionInstance(SPECS[VERIFY], target=str(row.get("evidence_id") or ""),
                frontier_target_id=str(row.get("question_id") or "goal:G0"),
                reason="check_exact_evidence_identity", extra={"verify_kind": VERIFY_REPLICATION,
                                                               "evidence_role": row.get("role")}))
        elif row.get("role") == "ProbeEvidence" and row.get("reproduction_ok") and not row.get("replication_of"):
            for world in state.world_versions:
                wid = str(world.get("id") or world.get("world_id") or "")
                if world.get("research_role") != "heldout" or wid == row.get("world_id"):
                    continue
                # A registered script names the file it reads, so two worlds can
                # only host the same experiment when they share a payload name.
                # Demanding a replication no script can perform left the
                # dimension permanently missing and spent a tick each time.
                if not _replication_compatible_ids(state, str(row.get("world_id") or ""), wid):
                    continue
                if any(r.get("replication_of") == row.get("evidence_id") and r.get("world_id") == wid for r in state.evidence_records):
                    continue
                found.append(ActionInstance(SPECS[VERIFY], target=str(row.get("evidence_id")),
                    frontier_target_id=str(row.get("question_id") or "goal:G0"), reason="heldout_replication_debt",
                    extra={"replication_world_id": wid, "evidence_role": "ProbeEvidence",
                           "study_stage": "replication"}))
    if not surveys:
        questions = state.open_questions + state.blocked_questions
        target = str(questions[0].get("id")) if questions else "goal:G0"
        found.append(ActionInstance(SPECS[SURVEY], target=target,
                                   frontier_target_id=target, reason="literature_gap",
                                   extra={"extract_trajectories": True}))
        return unattempted_actions(state, found)
    found.extend(repair_actions(state))
    questions = state.open_questions + state.blocked_questions + state.resolved_questions
    for question in questions:
        qid = str(question.get("id") or "")
        if not any(r.get("question_id") in {"", qid} for r in surveys):
            found.append(ActionInstance(SPECS[SURVEY], target=qid, frontier_target_id=qid,
                reason="question_literature_gap", extra={"extract_trajectories": True}))
            continue
        theories = [t for t in state.theories if t.get("question_id") == qid]
        # A grounded transition needs one retrieved paper to cite another. Keyword
        # retrieval rarely returns a citation-linked pair, so this library is
        # often empty — and offering a lane with nothing to ground it spends a
        # tick to fail in a way the state already predicts.
        grounded = any(r.get("research_transitions") for r in surveys)
        lanes = ("trajectory_supported", "assumption_reframe", "unconstrained") if world_bound else ()
        for lane in lanes:
            if lane == "trajectory_supported" and not grounded:
                continue
            if not any(t.get("research_lane") == lane and int(t.get("hop") or 1) == 1 for t in theories):
                found.append(ActionInstance(SPECS[THEORIZE], target=qid, frontier_target_id=qid,
                    lineage_id=str(question.get("lineage_id") or ""), reason="first_hop_branch_gap",
                    extra={"research_lane": lane, "hop": 1}))
        # A promising empirical discovery idea enters the evidence runtime here:
        # compiled, once, into a registered hypothesis on the root question.
        if question is questions[0]:
            for idea_id in bridge_candidates(state):
                found.append(ActionInstance(SPECS[THEORIZE], target=qid, frontier_target_id=qid,
                    lineage_id=str(question.get("lineage_id") or ""),
                    reason="compile_promising_discovery_idea",
                    extra={"research_lane": "discovery_bridge", "hop": 1,
                           "discovery_idea_id": idea_id}))
        for theory in theories:
            if theory.get("role") == "competing_explanation":
                continue
            tid = str(theory.get("id") or "")
            card = theory.get("card") or {}
            measured = any(r.get("theory_id") == tid for r in state.evidence_records)
            attempted = any(r.get("theory_id") == tid and r.get("action_type") == PROBE
                            for r in state.failed_designs)
            literature_checked = any(r.get("theory_id") == tid for r in surveys)
            if card and not literature_checked:
                found.append(ActionInstance(SPECS[SURVEY], target=qid, frontier_target_id=qid,
                    reason="hypothesis_literature_check", extra={"theory_id": tid, "claim": theory.get("claim") or card.get("claim")}))
            if (card and literature_checked and not measured and not attempted
                    and not _registered_experiment_pending(state, qid)):
                found.append(ActionInstance(SPECS[PROBE], target=tid, frontier_target_id=qid,
                    reason="discriminate_competing_hypotheses" if str(card.get("idea_kind") or "probe").lower() == "probe"
                           else "check_missing_probe_compilation", extra={
                        "theory_id": tid, "question_id": qid, "card": card,
                        "causal_handle": card.get("world_lever") or theory.get("mechanism"),
                        "measure": card.get("world_observable") or card.get("prediction"),
                        "freeze": state.world_id,
                        # Only attach a proposal when one exists: an absent
                        # proposal is a plain probe, a null one is a filter that
                        # silently removes every experiment from the menu.
                        **({"experiment_proposal": theory["experiment_proposal"]}
                           if isinstance(theory.get("experiment_proposal"), dict) else {}),
                    }))
            if measured:
                found.append(ActionInstance(SPECS[SYNTHESIZE], target=tid, frontier_target_id=qid,
                                           reason="synthesize_measured_branch"))
            evidence = [r for r in state.evidence_records if r.get("theory_id") == tid
                        and r.get("epistemic") == "WORLD" and r.get("attested")
                        and r.get("execution_status") == "ran" and r.get("world_digest")
                        and r.get("experiment_digest")]
            failures = [r for r in state.failed_designs if r.get("theory_id") == tid]
            has_child = any(t.get("parent_theory_id") == tid for t in theories)
            # A blocked method can be reframed once; it is never called an evidence hop.
            pivot = failures and int(theory.get("failure_pivots") or 0) < 1
            if not has_child and (evidence or pivot):
                found.append(ActionInstance(SPECS[THEORIZE], target=qid, frontier_target_id=qid,
                    lineage_id=str(question.get("lineage_id") or ""),
                    reason="evidence_driven_transition" if evidence else "failure_driven_reframe",
                    extra={"research_lane": "trajectory_supported" if evidence else "assumption_reframe",
                           "hop": int(theory.get("hop") or 1) + 1, "parent_theory_id": tid,
                           "transition_kind": "evidence_driven" if evidence else "failure_reframe",
                           "evidence_context": evidence, "failure_context": failures,
                           "failure_pivots": int(theory.get("failure_pivots") or 0) + (0 if evidence else 1)}))
    return unattempted_actions(state, found)


def _read_slice(state: ScientificState, name: str, action: ActionInstance) -> Any:
    """The part of the state one declared ``ActionSpec.reads`` entry refers to.

    Deliberately narrow. An action whose declared inputs have not changed will
    produce the same result it produced last time, so it must not be offered
    again — and an action whose inputs *have* changed must be.
    """
    if name == "literature":
        return [r.get("id") for r in state.artifacts if r.get("kind") == "literature_survey"]
    if name in {"scientific_archive", "evidence_records"}:
        return [(r.get("evidence_id"), r.get("verify_count"), r.get("outcome"))
                for r in state.evidence_records]
    if name in {"world_version", "world_lineage", "world_id"}:
        return [(r.get("id") or r.get("world_id"), r.get("digest"), r.get("role"))
                for r in state.world_versions] + [state.world_id]
    if name in {"capability_registry",}:
        return sorted(state.available_capabilities)
    if name == "missing_capabilities":
        return sorted(state.missing_capabilities)
    if name == "open_questions":
        # Identity only. A failing action records an ActionBlocked that changes
        # the question's status, and status in this slice let that action
        # unblock its own retry — the same doomed lane was rescheduled every
        # tick because its own failure looked like a changed input.
        return sorted(str(r.get("id") or "") for r in
                      state.open_questions + state.blocked_questions
                      + state.partial_questions + state.resolved_questions)
    if name in {"failure_archive", "failed_designs"}:
        return [(r.get("action_type"), r.get("theory_id"), r.get("reason"))
                for r in state.failed_designs]
    if name == "working_memory":
        # What an archiving or reflecting action would have to work on.
        return {"closed": [r.get("id") for r in state.resolved_questions],
                "failures": len(state.failed_designs),
                "anomalies": list(state.anomalies)}
    if name == "evidence_id":
        target = next((r for r in state.evidence_records
                       if str(r.get("evidence_id") or "") == str(action.target or "")), {})
        return (target.get("evidence_id"), target.get("verify_count"),
                target.get("reproduction_ok"), target.get("replication_of"),
                action.extra.get("replication_world_id"))
    if name in {"harness_probe_executor", "harness_version", "harness_archive"}:
        return state.harness_version
    if name == "reviews":
        return [(r.get("review_id"), r.get("status"), r.get("severity")) for r in state.reviews]
    if name == "quality_state":
        return {"gap": sorted((state.quality or {}).get("quality_gap") or []),
                "statuses": {k: v.get("status") for k, v in
                             sorted(((state.quality or {}).get("dimensions") or {}).items())}}
    if name == "research_contract":
        return (state.contract or {}).get("root_question", "")
    if name == "resources":
        return (state.resources or {}).get("exhausted")
    if name == "scientific_state":
        # Only for actions that genuinely branch on the whole project shape.
        return {"questions": [r.get("id") for r in state.open_questions],
                "theories": [(r.get("id"), r.get("status")) for r in state.theories],
                "capabilities": sorted(state.missing_capabilities)}
    if name == "discovery_archive":
        from .discovery import DISCOVERY_ARTIFACT_KINDS

        return [(r.get("kind"), r.get("id")) for r in state.artifacts
                if r.get("kind") in DISCOVERY_ARTIFACT_KINDS]
    if name == "exploration_policy":
        from .discovery import current_policy

        return current_policy(state).get("version")
    if name == "research_intent":
        return (state.contract or {}).get("root_question", "") or state.goal
    return name


# A repair type names what has to change; the transformation that changes it.
REPAIR_OPERATOR = {
    "experiment_redesign": "representation_change",
    "measurement_redesign": "verifier_shift",
    "causal_redesign": "mechanism_replacement",
    "stronger_baseline": "boundary_search",
    "broader_evaluation": "scaling",
    "assumption_reframe": "assumption_inversion",
    "representation_change": "representation_change",
}


def repair_actions(state: ScientificState) -> list[ActionInstance]:
    """Turn an open design revision into the hypothesis that carries its repair.

    A revision states what failed, what must change, and that only new WORLD
    evidence can settle it. Nothing converted that into an experiment: four
    revisions in one live segment produced no new hypothesis and no second
    probe, so the project could not act on its own redesign and the repair loop
    ended at a note. The revision now opens the obligation that produces the
    repaired design; the experiment itself follows the ordinary probe path.
    """
    epoch = int(state.project_epoch)
    attempted = {
        str(row.get("revision_id") or "")
        for row in state.theories if row.get("revision_id")
    }
    found: list[ActionInstance] = []
    for revision in state.design_revisions:
        rid = str(revision.get("revision_id") or "")
        if (not rid or rid in attempted
                or int(revision.get("project_epoch") or 0) != epoch
                or str(revision.get("outcome") or "open") != "open"):
            continue
        # Resolve the parent to a hypothesis that exists, or to none. A repair
        # aimed at a hypothesis id the state does not hold is refused downstream
        # as a lineage mismatch, which is how four live revisions produced no
        # experiment even after they reached the menu.
        # The parent the revision recorded, and only that. A fallback that
        # re-derived it from the newest hypothesis made this action's identity
        # change every tick, so the retry guard could never recognise it.
        named = str(revision.get("parent_design") or "")
        parent = next((row for row in state.theories
                       if named and str(row.get("id") or "") == named), {})
        if parent.get('status') in {'refuted', 'retired'}:
            continue
        parent_id = str(parent.get("id") or "")
        qid = str(parent.get("question_id") or "")
        if not qid:
            qid = next((str(row.get("id") or "") for row in
                        state.open_questions + state.blocked_questions), "")
        if not qid:
            continue
        repair = str(revision.get("selected_repair") or "experiment_redesign")
        found.append(ActionInstance(
            SPECS[THEORIZE], target=qid, frontier_target_id=qid,
            lineage_id=str(parent.get("lineage_id") or ""),
            reason="carry_out_design_revision",
            extra={
                "research_lane": "assumption_reframe",
                "hop": int(parent.get("hop") or 1) + 1 if parent else 1,
                "parent_theory_id": parent_id,
                "transition_kind": "design_repair",
                "revision_id": rid,
                "repair": repair,
                "repair_operator": REPAIR_OPERATOR.get(repair, "representation_change"),
                "changed_components": list(revision.get("changed_components") or []),
                "failure_root_cause": str(revision.get("failure_root_cause") or ""),
                "early_kill_condition": str(revision.get("early_kill_condition") or ""),
            }))
    return found


def _replication_compatible_ids(state: ScientificState, primary: str, heldout: str) -> bool:
    """Can these two recorded worlds host the same registered experiment?"""
    rows = {str(row.get("id") or row.get("world_id") or ""): dict(row)
            for row in state.world_versions}
    left, right = rows.get(primary) or {}, rows.get(heldout) or {}
    if not left or not right:
        return False
    if left.get("schema") and right.get("schema") and left["schema"] != right["schema"]:
        return False

    def payloads(row: Mapping[str, Any]) -> set[str]:
        from pathlib import Path as _Path

        skip = {"seed.json", "origin.json", "manifest.json"}
        return {_Path(str(name)).name for name in row.get("files") or []
                if _Path(str(name)).name not in skip}

    known = payloads(left), payloads(right)
    if not all(known):
        return True  # files not recorded on the row; let the worker decide
    return bool(known[0] & known[1])


def action_input_key(state: ScientificState, action: ActionInstance) -> str:
    """A retry needs changed *declared inputs*, not just a later clock tick.

    This used to hash the whole scientific state, which defeated its own
    purpose: any unrelated change — one new evidence row, one new artifact —
    made every previously-failed action look untried again, so a blocked
    OBSERVE or an ACQUIRE with no recipe was rescheduled indefinitely. The key
    is now built from ``ActionSpec.reads``, the inputs the action itself
    declares. If a spec under-declares its inputs that is a spec to fix, not a
    reason to hash the world.
    """
    import hashlib
    import json

    inputs = {
        "action": action.action_type,
        "target": action.target,
        "reads": {name: _read_slice(state, name, action)
                  for name in sorted(action.spec.reads)},
        "branch": {k: action.extra.get(k) for k in
                   ("research_lane", "parent_theory_id", "hop", "theory_id",
                    "replication_world_id")},
        "project": {
            "epoch": state.project_epoch,
            "demand": (action.extra.get("demand") or {}).get("review_id")
                      or (action.extra.get("demand") or {}).get("dimension") or "",
            "lanes": sorted(str(row.get("lane") or "")
                            for row in action.extra.get("fork_candidates") or []),
        },
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()


def unattempted_actions(state: ScientificState, actions: list[ActionInstance]) -> list[ActionInstance]:
    # A new invocation may retry a recovered service with identical scientific
    # inputs. Ordinary scientific failures remain suppressed across sessions.
    session_start = max((i for i, row in enumerate(state.artifacts)
                         if row.get('kind') == 'research_service_session'), default=-1)
    attempted = {r.get("input_key") for i, r in enumerate(state.artifacts)
                 if r.get("kind") == "action_attempt"
                 and (r.get('status') not in {'service_unavailable', 'budget_exhausted'} or i >= session_start)}
    # A recovery that has spent its attempts on the same substantive gap, under
    # the same admitted capabilities and worlds, is not untried — it is done.
    # Renaming the question does not buy it another go; an admitted capability
    # or a new attested world does, because those change what can be run.
    from .recovery import exhausted, stalled
    return [a for a in actions
            if action_input_key(state, a) not in attempted
            and not exhausted(state, a) and not stalled(state, a)]


def types_of(actions: Iterable[ActionInstance]) -> tuple[str, ...]:
    return tuple(item.action_type for item in actions)


ACTION_COMPLETENESS: dict[str, dict[str, bool]] = {
    SURVEY: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    ASK: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    OBSERVE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    ACQUIRE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    THEORIZE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    PROBE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    VERIFY: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": True,
        "reducer": True,
        "regression_test": True,
    },
    REFLECT: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    SYNTHESIZE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    ARCHIVE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": False,
    },
    FORK_LINEAGE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    EVOLVE_HARNESS: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    # Discovery. No typed evidence by construction; see test_discovery_runtime.
    IDEATE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    CRITIQUE: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
    EVOLVE_POLICY: {
        "eligibility": True,
        "planner": True,
        "executor": True,
        "validator": True,
        "typed_evidence": False,
        "reducer": True,
        "regression_test": True,
    },
}
