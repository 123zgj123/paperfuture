"""Research progress, stagnation, and trajectory-level multistart.

Four different things used to be called "progress":

``activity``               an action ran and did not crash
``operational progress``   an artifact or a state row appeared
``epistemic progress``     the project learned something about the world
``research progress``      the *central* question moved, or the project's
                           quality gap shrank, or a real objection was retired

Only the first two were measured. That is why an agent could solve ten
peripheral questions, add ten artifacts, even admit new WORLD evidence, and
never trip a stagnation check — every tick "made progress" while the research
project stood exactly still. The distinction below is the difference between an
agent that is busy and an agent that is getting somewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..validity import ratio
from .actions import FORK_LINEAGE, ActionInstance
from .contract import alignment_of_question, contract_from_state
from .quality import STATUS_RANK, open_reviews
from .state import ScientificState


LANES = (
    "exploration",
    "exploitation",
    "revival",
    "understanding",
    "infrastructure",
)

STAGNATION_IDLE = 4

# A research project may be busy for a long time before anyone notices it is not
# advancing. This is deliberately longer than the operational idle window: it
# fires on *strategic* failure, which needs a few ticks of evidence.
STRATEGIC_STAGNATION_IDLE = 6


@dataclass
class ResearchProgress:
    question_resolution: int = 0
    validated_new_evidence: int = 0
    capability_unlock: int = 0
    theory_discrimination: int = 0
    frontier_advancement: int = 0
    high_value_negative: int = 0
    archive_reuse: int = 0
    contradiction_resolution: int = 0
    unused_capability: int = 0
    equivalent_theory: int = 0
    duplicate_evidence: int = 0
    # Research-level gains. These ask whether the *project* moved, not whether
    # the runtime did something.
    central_question_progress: int = 0
    decision_relevant_evidence: int = 0
    quality_gap_reduction: int = 0
    critical_review_resolution: int = 0
    hypothesis_discrimination: int = 0
    replication_progress: int = 0
    evaluation_strength_progress: int = 0
    mechanism_progress: int = 0
    successful_reorientation: int = 0
    # Keep ordinal changes as diagnostics; only evidenced changes below earn
    # research progress. A rating alone does not establish that a gap closed.
    quality_dimension_progress: int = 0
    objection_discovery: int = 0
    failure_information: int = 0
    # Research-level costs. Activity that looks like work and is not.
    low_value_activity: int = 0
    duplicate_progress: int = 0
    peripheral_question_progress: int = 0
    caveat_only_repair: int = 0
    same_failure_repetition: int = 0
    quality_stagnation: int = 0
    # Discovery packets and favorable feedback are operational activity;
    # neither is evidence nor independently contributes research_gain.
    discovery_progress: int = 0
    idea_progress: int = 0

    @property
    def research_gain(self) -> int:
        return (
            self.central_question_progress
            + self.decision_relevant_evidence
            + self.critical_review_resolution
            + self.hypothesis_discrimination
            + self.replication_progress
        )

    @property
    def research_cost(self) -> int:
        return (
            self.low_value_activity
            + self.duplicate_progress
            + self.peripheral_question_progress
            + self.caveat_only_repair
            + self.same_failure_repetition
            + self.quality_stagnation
        )

    @property
    def research_meaningful(self) -> bool:
        """Did the research project advance, as opposed to the runtime?

        Peripheral questions, duplicate evidence and prose repairs do not clear
        this bar no matter how many of them a tick produced.
        """
        return self.research_gain > self.research_cost

    @property
    def new_evidence(self) -> int:
        return self.validated_new_evidence

    @property
    def frontier_improvement(self) -> int:
        return self.frontier_advancement

    @property
    def meaningful(self) -> bool:
        gained = (
            self.question_resolution
            + self.validated_new_evidence
            + self.capability_unlock
            + self.theory_discrimination
            + self.frontier_advancement
            + self.high_value_negative
            + self.archive_reuse
            + self.contradiction_resolution
            + self.discovery_progress
        )
        penalty = self.unused_capability + self.equivalent_theory + self.duplicate_evidence
        return gained > penalty

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_resolution": self.question_resolution,
            "validated_new_evidence": self.validated_new_evidence,
            "new_evidence": self.validated_new_evidence,
            "capability_unlock": self.capability_unlock,
            "theory_discrimination": self.theory_discrimination,
            "frontier_advancement": self.frontier_advancement,
            "frontier_improvement": self.frontier_advancement,
            "high_value_negative": self.high_value_negative,
            "archive_reuse": self.archive_reuse,
            "contradiction_resolution": self.contradiction_resolution,
            "unused_capability": self.unused_capability,
            "equivalent_theory": self.equivalent_theory,
            "duplicate_evidence": self.duplicate_evidence,
            "meaningful": self.meaningful,
            "central_question_progress": self.central_question_progress,
            "decision_relevant_evidence": self.decision_relevant_evidence,
            "quality_gap_reduction": self.quality_gap_reduction,
            "critical_review_resolution": self.critical_review_resolution,
            "hypothesis_discrimination": self.hypothesis_discrimination,
            "replication_progress": self.replication_progress,
            "evaluation_strength_progress": self.evaluation_strength_progress,
            "mechanism_progress": self.mechanism_progress,
            "successful_reorientation": self.successful_reorientation,
            "quality_dimension_progress": self.quality_dimension_progress,
            "objection_discovery": self.objection_discovery,
            "failure_information": self.failure_information,
            "low_value_activity": self.low_value_activity,
            "duplicate_progress": self.duplicate_progress,
            "peripheral_question_progress": self.peripheral_question_progress,
            "caveat_only_repair": self.caveat_only_repair,
            "same_failure_repetition": self.same_failure_repetition,
            "quality_stagnation": self.quality_stagnation,
            "discovery_progress": self.discovery_progress,
            "idea_progress": self.idea_progress,
            "research_gain": self.research_gain,
            "research_cost": self.research_cost,
            "research_meaningful": self.research_meaningful,
        }


def _discriminating(theories: list[dict[str, Any]]) -> int:
    n = 0
    for row in theories:
        preds = list(row.get("discriminating_predictions") or [])
        if any(item.get("competitor_differs") for item in preds if isinstance(item, Mapping)):
            n += 1
    return n


def _world_evidence(state: ScientificState) -> list[dict[str, Any]]:
    return [
        row
        for row in state.evidence_records
        if str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id")
    ]


def _quality_gap_size(state: ScientificState) -> int:
    quality = state.quality or {}
    if "quality_gap_size" in quality:
        return int(quality.get("quality_gap_size") or 0)
    from .quality import compute_quality
    return int(compute_quality(state).get("quality_gap_size") or 0)


def _dimension_rank(state: ScientificState, name: str) -> int:
    quality = state.quality or {}
    row = (quality.get("dimensions") or {}).get(name) or {}
    return STATUS_RANK.get(str(row.get("status") or "unknown"), 0)


def _central_ids(state: ScientificState) -> set[str]:
    """Questions that still serve the contract's root question."""
    contract = contract_from_state(state)
    if not contract.is_set:
        return {str(row.get("id") or "") for row in
                state.open_questions + state.resolved_questions + state.partial_questions}
    known = (state.open_questions + state.resolved_questions
             + state.partial_questions + state.blocked_questions)
    return {
        str(row.get("id") or "")
        for row in known
        if alignment_of_question(contract, row, known=known).status in {"aligned", "instrumental"}
    }


def _open_severity(state: ScientificState, severity: str) -> int:
    return len(open_reviews(state.reviews, severity=severity))


def _failure_signature(row: Mapping[str, Any]) -> str:
    return "|".join(str(row.get(key) or "") for key in
                    ("action_type", "theory_id", "reason"))


def measure_research_progress(
    before: ScientificState,
    after: ScientificState,
    basic: "ResearchProgress",
) -> "ResearchProgress":
    """Fill the research-level dimensions of an already-measured tick.

    Everything here is a difference between two canonical states. A worker
    cannot report that it made research progress; it can only leave behind a
    state in which the central question is better supported than before.
    """
    central_after = _central_ids(after)
    central_before = _central_ids(before)
    before_eids = {str(row.get("evidence_id") or "") for row in before.evidence_records}
    new_rows = [row for row in after.evidence_records
                if str(row.get("evidence_id") or "") not in before_eids]
    from ..validity import evidence_eligible
    admitted = [row for row in new_rows if evidence_eligible(row)]

    central_theories = {
        str(row.get("id") or "") for row in after.theories
        if row.get("role") != "competing_explanation"
        and (not central_after or str(row.get("question_id") or "") in central_after)
    }
    decision_relevant = [
        row for row in admitted
        if str(row.get("theory_id") or "") in central_theories
        or str(row.get("question_id") or "") in central_after
    ]
    peripheral = [row for row in admitted if row not in decision_relevant]

    resolved_central = [
        str(row.get("id") or "") for row in after.resolved_questions
        if str(row.get("id") or "") in central_after
        and not any(str(item.get("id") or "") == str(row.get("id") or "")
                    for item in before.resolved_questions)
    ]
    resolved_peripheral = max(
        0,
        (len(after.resolved_questions) - len(before.resolved_questions)) - len(resolved_central),
    )

    gap_before, gap_after = _quality_gap_size(before), _quality_gap_size(after)
    fatal_closed = max(0, _open_severity(before, "fatal") - _open_severity(after, "fatal"))
    major_closed = max(0, _open_severity(before, "major") - _open_severity(after, "major"))

    refused_before = int((before.metrics or {}).get("caveat_only_repair_refused") or 0)
    refused_after = int((after.metrics or {}).get("caveat_only_repair_refused") or 0)

    before_failures = [_failure_signature(row) for row in before.failed_designs]
    new_failures = [_failure_signature(row) for row in after.failed_designs[len(before.failed_designs):]]
    repeated = sum(1 for sig in new_failures if sig in before_failures)

    before_questions = {q.get('id') for q in before.open_questions + before.blocked_questions
                        + before.resolved_questions + before.partial_questions}
    reoriented = sum(1 for q in after.open_questions if q.get('id') not in before_questions
                     and q.get('parent_question_id') and q.get('transition_status') == 'active')

    belief_changes = sum(1 for tid, belief in after.belief_states.items()
        if tid in before.belief_states and any(belief.get(key) != before.belief_states[tid].get(key)
            for key in ('status', 'observation_status', 'supporting_evidence', 'contradicting_evidence')))
    basic.central_question_progress = len(resolved_central) + belief_changes
    basic.decision_relevant_evidence = len(decision_relevant)
    basic.quality_gap_reduction = max(0, gap_before - gap_after)
    basic.critical_review_resolution = fatal_closed * 2 + major_closed
    # A generated discriminating prediction is a proposal. Only observed
    # belief changes above establish scientific progress.
    basic.hypothesis_discrimination = 0
    basic.replication_progress = sum(1 for row in admitted if row.get("replication_of")
                                     and row.get("transfer_ok"))
    basic.evaluation_strength_progress = max(
        0,
        (_dimension_rank(after, "baseline_strength") - _dimension_rank(before, "baseline_strength"))
        + (_dimension_rank(after, "measurement_validity") - _dimension_rank(before, "measurement_validity")),
    )
    basic.mechanism_progress = max(
        0, _dimension_rank(after, "mechanism_support") - _dimension_rank(before, "mechanism_support"))
    basic.successful_reorientation = reoriented

    # Ordinal changes remain diagnostics, never research gain.
    basic.quality_dimension_progress = sum(
        max(0, _dimension_rank(after, name) - _dimension_rank(before, name))
        for name in ((after.quality or {}).get("dimensions") or {})
    )
    # Finding the flaw early is the capability under test. Structural review is
    # deduplicated and grounded in recorded state, so this cannot be farmed by
    # reviewing repeatedly.
    known_reviews = {str(row.get("review_id") or "") for row in before.reviews}
    basic.objection_discovery = sum(
        1 for row in after.reviews
        if str(row.get("review_id") or "") not in known_reviews
        and str(row.get("severity") or "") in {"fatal", "major"}
    )
    # Diagnostic activity only: a new worker error is not scientific evidence.
    # Verified negative observations already contribute through scoped beliefs
    # and admitted evidence, without awarding credit to an error string.
    basic.failure_information = max(0, len(new_failures) - repeated)

    basic.peripheral_question_progress = resolved_peripheral + len(peripheral)
    basic.caveat_only_repair = max(0, refused_after - refused_before)
    basic.same_failure_repetition = repeated
    basic.duplicate_progress = basic.duplicate_evidence
    basic.quality_stagnation = 1 if gap_after > gap_before else 0
    # An action that produced no admitted evidence, resolved no objection, moved
    # no quality dimension and opened no revision was activity, not research.
    basic.low_value_activity = 0 if (basic.research_gain or basic.meaningful) else 1
    return basic


def measure_progress(before: ScientificState, after: ScientificState) -> ResearchProgress:
    before_ids = {
        (str(row.get("evidence_id") or ""), str(row.get("role") or ""))
        for row in before.evidence_records
    }
    new_rows = [
        row
        for row in after.evidence_records
        if (str(row.get("evidence_id") or ""), str(row.get("role") or "")) not in before_ids
    ]
    validated = [
        row
        for row in new_rows
        if str(row.get("epistemic") or "") == "WORLD" and row.get("attested")
    ]
    duplicate = max(0, len(new_rows) - len({str(row.get("evidence_id") or "") for row in new_rows}))
    unused = [
        name
        for name in after.available_capabilities
        if name not in before.available_capabilities
        and not any(str(row.get("capability") or "") == name for row in after.evidence_records)
    ]
    basic = ResearchProgress(
        question_resolution=max(0, len(after.resolved_questions) - len(before.resolved_questions)),
        validated_new_evidence=len(validated),
        capability_unlock=max(0, len(after.available_capabilities) - len(before.available_capabilities)),
        theory_discrimination=max(0, _discriminating(after.theories) - _discriminating(before.theories)),
        frontier_advancement=max(
            0,
            len(after.resolved_questions) - len(before.resolved_questions),
        )
        + max(0, len(_world_evidence(after)) - len(_world_evidence(before))),
        high_value_negative=max(
            0,
            sum(1 for row in after.evidence_records if row.get("outcome") in {"negative", "weakens"})
            - sum(1 for row in before.evidence_records if row.get("outcome") in {"negative", "weakens"}),
        ),
        archive_reuse=max(0, len(after.archive_ids) - len(before.archive_ids)),
        contradiction_resolution=max(0, len(before.contradictions) - len(after.contradictions)),
        unused_capability=len(unused),
        equivalent_theory=max(
            0,
            (len(after.theories) - _discriminating(after.theories))
            - (len(before.theories) - _discriminating(before.theories)),
        ),
        duplicate_evidence=duplicate,
    )
    from .discovery import PROMISING, VERDICT_RANK, discovery_records

    ideas_before, feedback_before = discovery_records(before)
    ideas_after, feedback_after = discovery_records(after)
    basic.discovery_progress = (max(0, len(ideas_after) - len(ideas_before))
                                + max(0, len(feedback_after) - len(feedback_before)))
    parents = {str(row["id"]): str(row.get("parent_id") or "") for row in ideas_after}
    for idea_id, row in feedback_after.items():
        if idea_id in feedback_before:
            continue
        parent = feedback_after.get(parents.get(idea_id, ""))
        if row.get("verdict") == PROMISING or (parent is not None and VERDICT_RANK.get(
                str(row.get("verdict")), 0) > VERDICT_RANK.get(str(parent.get("verdict")), 0)):
            basic.idea_progress += 1
    return measure_research_progress(before, after, basic)


@dataclass
class StagnationEvent:
    idle_ticks: int
    lanes: tuple[str, ...] = LANES
    reason: str = "no_meaningful_progress"
    forbid: str = "far_jump_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "idle_ticks": self.idle_ticks,
            "lanes": list(self.lanes),
            "reason": self.reason,
            "forbid": self.forbid,
        }


def maybe_stagnation(state: ScientificState, *, idle: int = STAGNATION_IDLE) -> StagnationEvent | None:
    """Operational idleness: nothing at all is happening."""
    gap = int(state.ticks) - int(state.last_progress_tick)
    if gap >= idle and state.ticks > 0:
        return StagnationEvent(idle_ticks=gap)
    return None


@dataclass
class StrategicStagnation:
    """The project is busy and going nowhere.

    Every tick may succeed. Every tick may add an artifact. Some ticks may even
    admit new WORLD evidence. If the central question is no better supported,
    the major objections are not fewer and the quality gap has not shrunk, the
    research is stagnant and the answer is not another experiment of the same
    shape — it is REDESIGN, FORK, or PROJECT_PIVOT.
    """

    idle_ticks: int
    reasons: tuple[str, ...] = ()
    severity: str = "strategic"
    quality_gap: int = 0
    open_fatal: int = 0
    open_major: int = 0
    escalate: tuple[str, ...] = ("REDESIGN", "FORK_LINEAGE", "PROJECT_PIVOT")

    def to_dict(self) -> dict[str, Any]:
        return {
            "idle_ticks": self.idle_ticks,
            "reasons": list(self.reasons),
            "severity": self.severity,
            "quality_gap": self.quality_gap,
            "open_fatal": self.open_fatal,
            "open_major": self.open_major,
            "escalate": list(self.escalate),
            "basis": "no_research_level_progress_despite_operational_activity",
        }


def maybe_strategic_stagnation(
    state: ScientificState, *, idle: int = STRATEGIC_STAGNATION_IDLE
) -> StrategicStagnation | None:
    if int(state.ticks) <= 0:
        return None
    gap = int(state.ticks) - int(state.last_research_progress_tick)
    if gap < idle:
        return None
    reasons: list[str] = []
    quality_gap = _quality_gap_size(state)
    fatal, major = _open_severity(state, "fatal"), _open_severity(state, "major")
    if quality_gap:
        reasons.append(f"quality_gap_unchanged_for_{gap}_ticks:{quality_gap}_dimensions_short")
    if fatal:
        reasons.append(f"unresolved_fatal_objections:{fatal}")
    if major:
        reasons.append(f"unresolved_major_objections:{major}")
    if not state.verified_research_state:
        reasons.append("no_verified_central_claim")
    if int((state.metrics or {}).get("caveat_only_repair_refused") or 0):
        reasons.append("repairs_attempted_with_prose_instead_of_evidence")
    if int((state.debt or {}).get("goal_drift_debt") or 0):
        reasons.append("open_questions_have_drifted_from_the_root_question")
    if not reasons:
        # Nothing left to complain about and no research progress either: this
        # is exhaustion, not stagnation. Termination handles it.
        return None
    return StrategicStagnation(
        idle_ticks=gap, reasons=tuple(reasons), quality_gap=quality_gap,
        open_fatal=fatal, open_major=major,
    )


@dataclass
class TrajectoryBranch:
    lane: str
    harness_version: str
    next_action: str
    snapshot: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    world_evidence: list[dict[str, Any]] = field(default_factory=list)
    generated: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "harness_version": self.harness_version,
            "next_action": self.next_action,
            "score": self.score,
            "world_evidence": list(self.world_evidence),
            "generated": list(self.generated),
        }


def fork_trajectories(
    state: ScientificState,
    *,
    harness_version: str,
    n: int = 3,
    preferred_actions: tuple[str, ...] = (),
) -> list[TrajectoryBranch]:
    """Snapshot state + harness and open short lanes. Not N idea cards."""
    snapshot = state.to_dict()
    lanes = list(LANES[: max(1, int(n))])
    actions = list(preferred_actions) or [
        "SURVEY",
        "THEORIZE",
        "EVOLVE_HARNESS",
        "ACQUIRE",
        "REFLECT",
    ]
    branches: list[TrajectoryBranch] = []
    for index, lane in enumerate(lanes):
        action = actions[index % len(actions)]
        if lane == "infrastructure" and "EVOLVE_HARNESS" in actions:
            action = "EVOLVE_HARNESS"
        if lane == "understanding":
            action = "THEORIZE"
        branches.append(
            TrajectoryBranch(
                lane=lane,
                harness_version=harness_version,
                next_action=action,
                snapshot=snapshot,
            )
        )
    return branches


def judge_branches(branches: list[TrajectoryBranch]) -> TrajectoryBranch:
    ranked = sorted(branches, key=lambda row: (-row.score, row.lane))
    return ranked[0]


def _evidence_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("question_id") or row.get("question") or row.get("target") or ""),
        str(row.get("claim") or row.get("contrast") or row.get("result") or ""),
        str(row.get("verdict") or row.get("outcome") or row.get("theory_relation") or ""),
    )


def classify_branch_evidence(
    rows_a: list[Mapping[str, Any]],
    rows_b: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """compatible / independent / conflicting. GENERATED never becomes WORLD."""
    compatible: list[dict[str, Any]] = []
    independent: list[dict[str, Any]] = []
    conflicting: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in list(rows_a) + list(rows_b):
        payload = dict(row)
        if str(payload.get("epistemic") or "") == "GENERATED":
            continue
        key = _evidence_key(payload)
        if key in seen:
            continue
        seen.add(key)
        match = None
        pool = rows_b if row in rows_a else rows_a
        for other in pool:
            other_key = _evidence_key(other)
            if other_key[0] == key[0] and other_key[0]:
                if other_key[2] and key[2] and other_key[2] != key[2]:
                    match = "conflict"
                    break
                if other_key == key:
                    match = "compatible"
                    break
        if match == "conflict":
            conflicting.append(payload)
        elif match == "compatible":
            compatible.append(payload)
        else:
            independent.append(payload)
    return {
        "compatible": compatible,
        "independent": independent,
        "conflicting": conflicting,
    }


def merge_branches(
    branches: list[TrajectoryBranch],
    state: ScientificState,
) -> dict[str, Any]:
    """Preserve all valid WORLD evidence. Score does not discard a branch."""
    world_rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for branch in branches:
        for row in branch.world_evidence:
            if str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id"):
                world_rows.append(dict(row))
        for row in branch.generated:
            generated.append(dict(row))
    half = max(1, len(world_rows) // 2)
    split = classify_branch_evidence(world_rows[:half], world_rows[half:])
    if len(branches) >= 2:
        split = classify_branch_evidence(
            [dict(row) for row in branches[0].world_evidence],
            [dict(row) for row in branches[1].world_evidence],
        )
    merged: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    for row in split["compatible"] + split["independent"]:
        if str(row.get("epistemic") or "") != "WORLD":
            continue
        if not any(
            item.get("evidence_id") == row.get("evidence_id") for item in state.evidence_records
        ):
            state.record_evidence(row)
        merged.append(row)
    for row in split["conflicting"]:
        if str(row.get("epistemic") or "") != "WORLD":
            continue
        if not any(
            item.get("evidence_id") == row.get("evidence_id") for item in state.evidence_records
        ):
            state.record_evidence(row)
        merged.append(row)
        contradictions.append(
            {
                "question_id": str(row.get("question_id") or row.get("question") or ""),
                "evidence_id": str(row.get("evidence_id") or ""),
                "theory_id": str(row.get("theory_id") or ""),
                "verdict": str(row.get("verdict") or row.get("theory_relation") or ""),
            }
        )
        if contradictions[-1] not in state.contradictions:
            state.contradictions.append(dict(contradictions[-1]))
    return {
        "merged": merged,
        "contradictions": contradictions,
        "generated_kept_local": generated,
        "promoted_generated": False,
    }


def merge_branch(winner: TrajectoryBranch, state: ScientificState) -> list[dict[str, Any]]:
    """Backward-compatible wrapper. Prefer merge_branches for conflicts."""
    result = merge_branches([winner], state)
    return list(result["merged"])


def default_metrics() -> dict[str, Any]:
    return {
        "action_diversity": [],
        "non_scripted_transitions": 0,
        "scripted_transitions": 0,
        "question_resolution_rate": None,
        "capability_growth": 0,
        "unlock_rate": 0,
        "harness_generalization": 0,
        "regression_rate": 0,
        "archive_reuse": 0,
        "lineage_progress": 0,
        "epistemic_integrity": 0,
    }


def update_metrics(
    state: ScientificState,
    *,
    action: ActionInstance | None = None,
    action_type: str = "",
    scripted: bool = False,
    unlock: bool = False,
    integrity_gap: str = "",
    archive_hit: bool = False,
    harness_general: bool = False,
    regression: bool = False,
) -> None:
    metrics = state.metrics if state.metrics else default_metrics()
    kind = (action.action_type if action is not None else "") or action_type
    if kind:
        seen = list(metrics.get("action_diversity") or [])
        if kind not in seen:
            seen.append(kind)
        metrics["action_diversity"] = seen
    if scripted:
        metrics["scripted_transitions"] = int(metrics.get("scripted_transitions") or 0) + 1
    else:
        metrics["non_scripted_transitions"] = int(metrics.get("non_scripted_transitions") or 0) + 1
    # A rate over nothing is undefined, not zero and not one.
    metrics["question_resolution_rate"] = ratio(
        len(state.resolved_questions), len(state.open_questions) + len(state.resolved_questions))
    metrics["capability_growth"] = len(state.available_capabilities)
    if unlock:
        metrics["unlock_rate"] = int(metrics.get("unlock_rate") or 0) + 1
    if archive_hit:
        metrics["archive_reuse"] = int(metrics.get("archive_reuse") or 0) + 1
    if harness_general:
        metrics["harness_generalization"] = int(metrics.get("harness_generalization") or 0) + 1
    if regression:
        metrics["regression_rate"] = int(metrics.get("regression_rate") or 0) + 1
    if integrity_gap:
        metrics["epistemic_integrity"] = int(metrics.get("epistemic_integrity") or 0) + 1
    else:
        metrics.setdefault("epistemic_integrity", 0)
    if state.resolved_questions and state.evidence_records:
        metrics["lineage_progress"] = int(metrics.get("lineage_progress") or 0) + 1
    world_ev = [
        row
        for row in state.evidence_records
        if str(row.get("epistemic") or "") == "WORLD" and row.get("evidence_id")
    ]
    verified = [row for row in world_ev if int(row.get("verify_count") or 0) >= 1]
    metrics["verified_evidence_yield"] = len(verified)
    metrics["theory_discrimination_rate"] = ratio(_discriminating(state.theories), len(state.theories))
    metrics["verification_depth"] = sum(int(row.get("verify_count") or 0) for row in world_ev)
    metrics["research_debt"] = dict(state.debt)
    # A resolution rate needs every contradiction ever raised, and only the open
    # ones are recorded; it read 1.0 when none had been raised at all. What the
    # state does know is how many stand open.
    metrics.pop("contradiction_resolution_rate", None)
    metrics["open_contradictions"] = len(state.contradictions)
    # Drift is recorded only on scientific-action ticks, so those ticks are its
    # denominator — the same set, counted before dividing, this tick included.
    # Dividing by the count from before this tick put the rate above 1.
    from .frontier import SCIENTIFIC_ACTIONS

    if kind in SCIENTIFIC_ACTIONS:
        metrics["scientific_actions"] = int(metrics.get("scientific_actions") or 0) + 1
    metrics["goal_drift_rate"] = ratio(int(metrics.get("goal_drift") or 0),
                                       int(metrics.get("scientific_actions") or 0))
    metrics.setdefault("meaningful_events", 0)
    state.metrics = metrics


def should_terminate(
    state: ScientificState,
    *,
    horizon: int,
    min_expected_value: float = 0.15,
) -> str | None:
    """Hard budget is a safety cap. Scientific exhaustion can stop earlier.

    ``horizon`` is an absolute tick count — the caller's cap, counted from the
    ticks the project already has.
    """
    # A model service with a spent balance ends the run whatever else is left;
    # every further tick would rediscover it.
    if (getattr(state, "resources", None) or {}).get("service_exhausted"):
        return "service_unavailable"
    from .termination import (readiness, BLOCKED, JUDGED_BELOW_CONTRACT, MISSION_NOT_DECLARED,
                              NO_VIABLE_PATH)
    verdict = readiness(state)
    if verdict.status in {MISSION_NOT_DECLARED, JUDGED_BELOW_CONTRACT}:
        # No action inside the loop can supply a mission definition, so there is
        # no recovery to try first. Continuing only spends the horizon, which is
        # what four live missions did after their science was finished.
        return verdict.status.lower()
    if verdict.status in {BLOCKED, NO_VIABLE_PATH}:
        from .actions import eligible_actions, unattempted_actions
        from .scheduler import admissible
        candidates, _ = admissible(state, unattempted_actions(state, eligible_actions(state, discovery=True)))
        recovery = [a for a in candidates if a.action_type in {'ACQUIRE', 'VERIFY', 'PROJECT_PIVOT'}]
        if not recovery:
            return verdict.status.lower()
    if int(state.ticks) >= int(horizon):
        return "budget"
    debt = state.debt or {}
    open_high = [
        row
        for row in state.open_questions
        if row.get("priority") in {None, "", "high"} or not row.get("priority")
    ]
    if (
        not open_high
        and not state.blocked_questions
        and int(debt.get("verification_debt") or 0) == 0
        and int(debt.get("contradiction_debt") or 0) == 0
        and state.resolved_questions
    ):
        return "frontier_exhausted"
    idle = int(state.ticks) - int(state.last_progress_tick)
    if idle >= STAGNATION_IDLE * 3 and state.resolved_questions and min_expected_value >= 0:
        return "stagnation"
    return None


def fork_action_reason() -> str:
    return FORK_LINEAGE
