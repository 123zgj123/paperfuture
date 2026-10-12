"""Discovery Runtime: complete ideas, validated, evolved by their feedback.

FarField runs two runtimes under one controller. This module is the first.

    Research Intent → REMOTE_BOOTSTRAP → Complete Idea → Validate → Feedback
        → Idea Evolution (research operator) → ... → Research Plan
        → Experiment Design,  with REMOTE_ESCAPE when a lineage is stuck and
        BRANCH_DIVERSIFICATION when alternatives are worth a lineage of their own.

The Evidence Runtime (THEORIZE → PROBE → VERIFY → Trusted Kernel → WORLD) is
the second, and it is untouched by anything here. Principles:

1. Remote search discovers a research basin; it does not define every
   subsequent move.
2. After the first complete idea, the fundamental search unit is the research
   hypothesis, not the latent point.
3. Validation failure is search information, not discarded work.
4. OpenWorld predicts how a research direction may develop; it does not
   certify scientific truth.
5. Discovery may be speculative. Evidence promotion must remain conservative.
6. A research idea can be worth pursuing before it is empirically proven.
7. FarField should optimize research progress, not conceptual distance.

What that means in code:

- Discovery records are `ArtifactAdded` rows of three kinds (ideas, feedback
  packets, exploration-policy updates). The kernel stamps them GENERATED. They
  never enter `theories`, `evidence_records`, belief bookkeeping or the quality
  projection, so nothing about them can be believed, promoted or counted as
  evidence, and no evidence gate can decide whether one may exist.
- Everything else here is derived from those rows on demand (`discovery_view`),
  so a state rebuilt from the event log yields the same view.
- An idea's research promise and its evidence status are separate fields. The
  first comes from discovery; the second is read, never written, from the
  evidence the kernel admitted.
- No function here reads a conceptual distance to rank, accept or reject an
  idea. Distances appear only as provenance on remote-search records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping, Sequence

IDEATE = "IDEATE"
CRITIQUE = "CRITIQUE"
EVOLVE_POLICY = "EVOLVE_POLICY"
DISCOVERY_ACTIONS = frozenset({IDEATE, CRITIQUE, EVOLVE_POLICY})

IDEA_KIND = "discovery_idea"
FEEDBACK_KIND = "idea_feedback"
POLICY_KIND = "exploration_policy"
DISCOVERY_ARTIFACT_KINDS = frozenset({IDEA_KIND, FEEDBACK_KIND, POLICY_KIND, 'discovery_decision'})

# How an idea came to be. Only the first two and the remote branch search.
REMOTE_BOOTSTRAP = "REMOTE_BOOTSTRAP"
EVOLVE = "EVOLVE"
REMOTE_ESCAPE = "REMOTE_ESCAPE"
BRANCH_DIVERSIFICATION = "BRANCH_DIVERSIFICATION"
MODES = (REMOTE_BOOTSTRAP, EVOLVE, REMOTE_ESCAPE, BRANCH_DIVERSIFICATION)
REMOTE_BRANCH = "remote_reconceptualization"

OPERATORS: dict[str, str] = {
    "MECHANISM_REPAIR": (
        "Keep the research problem and the phenomenon; replace the mechanism the "
        "feedback found unsupported with one that could actually produce it."),
    "ASSUMPTION_REMOVAL": (
        "Delete or weaken the assumption that makes the idea unrealistic or fragile, "
        "and say what the idea claims without it."),
    "REFORMULATE": (
        "Redefine the research problem so the phenomenon that is actually valuable is "
        "stated clearly; the old framing may be what failed."),
    "DISCRIMINATIVE_TEST": (
        "The idea and its strongest competing explanation currently predict the same "
        "thing. Restructure the claim and its test so the two predict different "
        "outcomes on the same measurement."),
    "MEASUREMENT_REPAIR": (
        "The proxy does not answer the claim. Replace it with a measurement whose value "
        "would differ depending on whether the claim is true."),
    "SCOPE_SHIFT": (
        "Narrow or widen the claim until it is testable, meaningful and not trivial."),
    "TRANSFER_STRUCTURE": (
        "Borrow a structure, mechanism or abstraction proven in another field and state "
        "the structural correspondence explicitly. Never splice in a distant concept "
        "without a correspondence."),
    "REMOTE_ESCAPE": (
        "The lineage is stuck. Keep the research problem and what survived; leave the "
        "lineage's conceptual frame through a new external frame."),
    "BRANCH_DIVERSIFICATION": (
        "Open a deliberately different research lineage from a current idea."),
}
BRANCH_KINDS = (REMOTE_BRANCH, "MECHANISM_REPAIR", "MEASUREMENT_REPAIR", "ASSUMPTION_REMOVAL")

NO_FAILURE = "none"
FAILURE_MODES = (
    "incomplete_idea",
    "unfalsifiable",
    "mechanism_unsupported",
    "measurement_invalid",
    "competing_explanation_unresolved",
    "unrealistic_assumption",
    "low_novelty",
    "scope_mismatch",
    "ill_posed_problem",
    "missing_structure",
    "unclassified",
)
# What a critic tends to call a failure, mapped onto the vocabulary above.
_FAILURE_SYNONYMS = (
    ("causal_identifiab", "competing_explanation_unresolved"),
    ("confound", "competing_explanation_unresolved"),
    ("competing", "competing_explanation_unresolved"),
    ("alternative", "competing_explanation_unresolved"),
    ("measure", "measurement_invalid"),
    ("proxy", "measurement_invalid"),
    ("mechanism", "mechanism_unsupported"),
    ("assumption", "unrealistic_assumption"),
    ("unrealistic", "unrealistic_assumption"),
    ("novel", "low_novelty"),
    ("prior", "low_novelty"),
    ("known", "low_novelty"),
    ("scope", "scope_mismatch"),
    ("trivial", "scope_mismatch"),
    ("broad", "scope_mismatch"),
    ("ill_posed", "ill_posed_problem"),
    ("framing", "ill_posed_problem"),
    ("problem", "ill_posed_problem"),
    ("structure", "missing_structure"),
    ("falsif", "unfalsifiable"),
    ("incomplete", "incomplete_idea"),
)

PROMISING, REVISE, COLLAPSE = "PROMISING", "REVISE", "COLLAPSE"
VERDICTS = (PROMISING, REVISE, COLLAPSE)
VERDICT_RANK = {COLLAPSE: 0, REVISE: 1, PROMISING: 2}
DEVELOPS, FRAGILE, COLLAPSES = "DEVELOPS", "FRAGILE", "COLLAPSES"
OUTCOMES = (DEVELOPS, FRAGILE, COLLAPSES)
PROMISE_LEVELS = ("HIGH", "MEDIUM", "LOW", "UNASSESSED")
PROMISE_RANK = {"UNASSESSED": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

EVIDENCE_STATUSES = (
    "UNVALIDATED",              # empirical claim, not yet taken into the evidence runtime
    "NOT_EMPIRICAL",            # conceptual or theoretical claim; V4 does not apply
    "AWAITING_ATTESTED_WORLD",  # empirical claim, but no attested world is bound
    "EVIDENCE_PENDING",         # compiled into a registered hypothesis, nothing admitted yet
    "EMPIRICALLY_SUPPORTED",
    "EMPIRICALLY_WEAKENED",
    "INCONCLUSIVE",
)
CLAIM_TYPES = ("conceptual", "theoretical", "empirical_observational", "mechanistic_intervention")
EMPIRICAL_CLAIMS = frozenset({"empirical_observational", "mechanistic_intervention"})

DISCOVERY_SUCCEEDED = "DISCOVERY_SUCCEEDED"
DISCOVERY_PARTIAL = "DISCOVERY_PARTIAL"
DISCOVERY_INCOMPLETE = "DISCOVERY_INCOMPLETE"
DISCOVERY_NOT_STARTED = "DISCOVERY_NOT_STARTED"

IDEA_FIELDS = (
    "research_problem",
    "central_claim",
    "mechanism",
    "motivation",
    "why_it_may_matter",
    "novelty_hypothesis",
    "competing_explanation",
    "falsifier",
    "validation_direction",
    "grounding",
    "uncertainties",
)
CORE_FIELDS = ("research_problem", "central_claim", "mechanism")


# ---------------------------------------------------------------------------
# The complete idea
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompleteIdea:
    """The unit discovery searches over after the first hop.

    Not a concept pair. An idea that cannot state its problem, claim and
    mechanism does not exist; one missing any other field exists and is
    flagged, because an incomplete idea is still something the next hop can
    repair.
    """

    research_problem: str = ""
    central_claim: str = ""
    mechanism: str = ""
    motivation: str = ""
    why_it_may_matter: str = ""
    novelty_hypothesis: str = ""
    competing_explanation: str = ""
    falsifier: str = ""
    validation_direction: str = ""
    optional_experiment_direction: str = ""
    grounding: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    title: str = ""
    claim_type: str = ""
    structural_correspondence: str = ""
    preserved_from_parent: tuple[str, ...] = ()
    changed_from_parent: tuple[str, ...] = ()
    addresses: str = ""
    experiment: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    design_candidates: list[dict[str, Any]] = field(default_factory=list)
    opportunity: dict[str, Any] = field(default_factory=dict)
    transfer_analysis: dict[str, Any] = field(default_factory=dict)
    exploration: dict[str, Any] = field(default_factory=dict)
    knowledge_request: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "CompleteIdea":
        row = dict(payload or {})

        def text(*names: str) -> str:
            for name in names:
                value = row.get(name)
                if isinstance(value, (list, tuple)):
                    value = "; ".join(str(item).strip() for item in value if str(item).strip())
                elif isinstance(value, Mapping):
                    value = "; ".join(f"{k}: {v}" for k, v in value.items())
                value = " ".join(str(value or "").split())
                if value:
                    return value
            return ""

        def items(name: str) -> tuple[str, ...]:
            value = row.get(name)
            if value is None:
                return ()
            if isinstance(value, str):
                value = [value]
            elif isinstance(value, Mapping):
                value = [f"{k}: {v}" for k, v in value.items()]
            out: list[str] = []
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(item, Mapping):
                    item = "; ".join(f"{k}: {v}" for k, v in item.items())
                item = " ".join(str(item or "").split())
                if item and item not in out:
                    out.append(item)
            return tuple(out)

        claim_type = text("claim_type").lower().replace("-", "_").replace(" ", "_")
        return cls(
            research_problem=text("research_problem", "problem"),
            central_claim=text("central_claim", "claim"),
            mechanism=text("mechanism"),
            motivation=text("motivation"),
            why_it_may_matter=text("why_it_may_matter", "importance"),
            novelty_hypothesis=text("novelty_hypothesis", "novelty"),
            competing_explanation=text("competing_explanation", "strongest_competing_explanation"),
            falsifier=text("falsifier"),
            validation_direction=text("validation_direction"),
            optional_experiment_direction=text("optional_experiment_direction", "experiment_direction"),
            grounding=items("grounding"),
            uncertainties=items("uncertainties"),
            title=text("title"),
            claim_type=claim_type if claim_type in CLAIM_TYPES else "",
            structural_correspondence=text("structural_correspondence"),
            preserved_from_parent=items("preserved_from_parent"),
            changed_from_parent=items("changed_from_parent"),
            addresses=text("addresses"),
            experiment=copy.deepcopy(dict(row['experiment'])) if isinstance(row.get('experiment'), Mapping) else {},
            predictions=copy.deepcopy(dict(row['predictions'])) if isinstance(row.get('predictions'), Mapping) else {},
            design_candidates=copy.deepcopy(row['design_candidates']) if isinstance(row.get('design_candidates'), list) else [],
            opportunity=copy.deepcopy(dict(row['opportunity'])) if isinstance(row.get('opportunity'), Mapping) else {},
            transfer_analysis=copy.deepcopy(dict(row['transfer_analysis'])) if isinstance(row.get('transfer_analysis'), Mapping) else {},
            exploration=copy.deepcopy(dict(row['exploration'])) if isinstance(row.get('exploration'), Mapping) else {},
            knowledge_request=copy.deepcopy(dict(row['knowledge_request'])) if isinstance(row.get('knowledge_request'), Mapping) else {},
        )

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name in IDEA_FIELDS if not getattr(self, name))

    def missing_core(self) -> tuple[str, ...]:
        return tuple(name for name in CORE_FIELDS if not getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            out[item.name] = list(value) if isinstance(value, tuple) else value
        return out


# ---------------------------------------------------------------------------
# Small helpers over a state that may be a ScientificState or its dict view
# ---------------------------------------------------------------------------


def _get(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _artifacts(state: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in (_get(state, "artifacts", []) or []) if isinstance(row, Mapping)]


def root_question(state: Any) -> str:
    contract = _get(state, "contract", {}) or {}
    return str(contract.get("root_question") or _get(state, "goal", "") or "")


_STOP = frozenset("""this that with from into than then they them their there these those which
while where when what whose would could should about above after again against being
because before between both does doing during each further have having here more most
other over same some such only very will your also just onto upon within without across
idea claim mechanism research problem""".split())


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9]+", str(text or "").lower())
            if len(word) > 3 and word not in _STOP}


def similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The exploration policy: P(operator | failure mode), versioned and installable
# ---------------------------------------------------------------------------

DEFAULT_POLICY: dict[str, Any] = {
    "version": "exploration-policy-v1",
    "parent_version": "",
    # Ranked operators per root failure cause. The first not already spent on
    # the same failure in the same lineage is chosen.
    "operator_for_failure": {
        "incomplete_idea": ["REFORMULATE", "SCOPE_SHIFT"],
        "unfalsifiable": ["DISCRIMINATIVE_TEST", "SCOPE_SHIFT"],
        "mechanism_unsupported": ["MECHANISM_REPAIR", "REFORMULATE", "TRANSFER_STRUCTURE"],
        "measurement_invalid": ["MEASUREMENT_REPAIR", "DISCRIMINATIVE_TEST"],
        "competing_explanation_unresolved": ["DISCRIMINATIVE_TEST", "MECHANISM_REPAIR"],
        "unrealistic_assumption": ["ASSUMPTION_REMOVAL", "SCOPE_SHIFT"],
        "low_novelty": ["REMOTE_ESCAPE", "TRANSFER_STRUCTURE"],
        "scope_mismatch": ["SCOPE_SHIFT", "REFORMULATE"],
        "ill_posed_problem": ["REFORMULATE", "SCOPE_SHIFT"],
        "missing_structure": ["TRANSFER_STRUCTURE", "MECHANISM_REPAIR"],
        "unclassified": ["REFORMULATE", "MECHANISM_REPAIR"],
    },
    "stagnation": {
        # The same root failure this many times in a row, never promising.
        "same_failure_streak": 3,
        # Consecutive ideas whose claim and mechanism barely changed.
        "structure_similarity": 0.85,
        "similar_streak": 2,
    },
    "remote": {
        "escape_on_stagnation": True,
        "max_escapes_per_lineage": 1,
        "branch_remote_first": True,
        "frames_per_jump": 3,
    },
    "limits": {
        "max_hops_per_lineage": 4,
        "max_lineages": 3,
        "max_ideas": 8,
        "branch_after_promising": True,
    },
}
POLICY_SECTIONS = ("operator_for_failure", "stagnation", "remote", "limits")
# Surfaces an exploration policy may never name. Discovery can evolve; the
# evidence kernel stays conservative (principle 5).
KERNEL_SURFACES = ("kernel", "promotion", "world", "validity", "evidence", "provenance",
                   "reducer", "readiness", "attest", "event")


def validate_policy(candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """May this be installed? Scope first, then shape."""
    reasons: list[str] = []
    row = dict(candidate or {})
    for key in row:
        lowered = str(key).lower()
        if any(surface in lowered for surface in KERNEL_SURFACES):
            reasons.append(f"{key}: an exploration policy may not touch the evidence kernel")
        elif key not in ("version", "parent_version", *POLICY_SECTIONS):
            reasons.append(f"{key}: not an exploration-policy section")
    mapping = row.get("operator_for_failure")
    if not isinstance(mapping, Mapping):
        reasons.append("operator_for_failure must map failure modes to ranked operators")
    else:
        for failure, ranked in mapping.items():
            if failure not in FAILURE_MODES:
                reasons.append(f"{failure}: unknown failure mode")
            if (not isinstance(ranked, list) or not ranked
                    or any(op not in OPERATORS for op in ranked)):
                reasons.append(f"{failure}: operators must be a non-empty list from the vocabulary")
    for section, bounds in (("stagnation", (0, 20)), ("limits", (1, 50)), ("remote", (0, 10))):
        body = row.get(section)
        if not isinstance(body, Mapping):
            reasons.append(f"{section} must be an object")
            continue
        for key, value in body.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not bounds[0] <= float(value) <= bounds[1]:
                reasons.append(f"{section}.{key}={value!r} is out of range")
    return (not reasons, reasons)


def policy_records(state: Any) -> list[dict[str, Any]]:
    return [row for row in _artifacts(state) if row.get("kind") == POLICY_KIND
            and int(row.get('project_epoch') or 0) == int(_get(state, 'project_epoch', 0) or 0)
            and row.get('status') != 'retired']


def scoped_policy_records(state: Any, lineage_id: str) -> list[dict[str, Any]]:
    sole = policy_lineage(state)
    return [row for row in policy_records(state)
            if (row.get('lineage_id') == lineage_id if row.get('lineage_id')
                else bool(sole and sole == lineage_id))]


def policy_lineage(state: Any, lineage_id: str = '') -> str:
    if lineage_id:
        return lineage_id
    ids = {str(r.get('lineage_id') or 'L1') for r in _artifacts(state)
           if r.get('kind') == IDEA_KIND and r.get('status') != 'retired'
           and int(r.get('project_epoch') or 0) == int(_get(state, 'project_epoch', 0) or 0)}
    return next(iter(ids)) if len(ids) == 1 else ''


def current_policy(state: Any, lineage_id: str = '') -> dict[str, Any]:
    scope = policy_lineage(state, lineage_id)
    installed = [row for row in scoped_policy_records(state, scope)
                 if row.get("installed") and isinstance(row.get("policy"), Mapping)]
    if installed:
        return copy.deepcopy(dict(installed[-1]["policy"]))
    return copy.deepcopy(DEFAULT_POLICY)


# ---------------------------------------------------------------------------
# The derived view
# ---------------------------------------------------------------------------


def discovery_records(state: Any, *, include_retired: bool = False) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ideas: list[dict[str, Any]] = []
    feedback: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in _artifacts(state):
        if not include_retired and (row.get('status') == 'retired'
                or int(row.get('project_epoch') or 0) != int(_get(state, 'project_epoch', 0) or 0)):
            continue
        kind = row.get("kind")
        if kind == IDEA_KIND and row.get("id") and row["id"] not in seen:
            seen.add(row["id"])
            ideas.append(row)
        elif kind == FEEDBACK_KIND and row.get("idea_id"):
            feedback.setdefault(str(row["idea_id"]), row)
    return ideas, {key: value for key, value in feedback.items() if key in seen}


def idea_of(row: Mapping[str, Any]) -> CompleteIdea:
    return CompleteIdea.from_payload(dict(row.get("idea") or {}))


def evidence_world_bound(state: Any) -> bool:
    """Is an attested world bound, so the evidence runtime has something to run on?

    A generated seed world is not: no registered experiment can execute against
    it. A world named without a recorded row is treated as bound, because the
    callers that do that bind a fixture directly.
    """
    wid = str(_get(state, "world_id", "") or "")
    if not wid:
        return False
    row = next((dict(item) for item in _get(state, "world_versions", []) or []
                if str(item.get("id") or item.get("world_id") or "") == wid), None)
    if row is None:
        return True
    if row.get("attested") is False or str(row.get("provenance") or "") == "generated":
        return False
    return True


def evidence_route(idea: CompleteIdea, state: Any) -> str:
    """Which validation an idea's claim calls for — V4 only when it is empirical."""
    if idea.claim_type not in EMPIRICAL_CLAIMS:
        return "NOT_APPLICABLE"
    return "V4_EVIDENCE_RUNTIME" if evidence_world_bound(state) else "AWAITING_ATTESTED_WORLD"


def evidence_status(state: Any, idea_row: Mapping[str, Any]) -> str:
    """Read from admitted evidence. Discovery never writes this."""
    idea_id = str(idea_row.get("id") or "")
    theories = {str(row.get("id") or "") for row in _get(state, "theories", []) or []
                if str(row.get("discovery_idea_id") or "") == idea_id
                and row.get("role") != "competing_explanation"}
    computational = [row for row in _get(state, 'evidence_records', []) or []
                     if row.get('discovery_idea_id') == idea_id
                     and row.get('study_stage') in {'confirmatory', 'replication'}
                     and row.get('registration_digest')]
    if theories or computational:
        from ..validity import corroborates

        rows = [row for row in _get(state, "evidence_records", []) or []
                if (str(row.get("theory_id") or "") in theories or row in computational)
                and str(row.get("epistemic") or "") == "WORLD" and row.get("promotion_ok")]
        if not rows:
            return "EVIDENCE_PENDING"
        outcomes = {str(row.get("outcome") or row.get("verdict") or "") for row in rows}
        if any(str(row.get("outcome") or "") == "supports" and corroborates(row) for row in rows):
            return "EMPIRICALLY_SUPPORTED"
        if outcomes & {"weakens", "negative", "contradicted"}:
            return "EMPIRICALLY_WEAKENED"
        return "INCONCLUSIVE"
    idea = idea_of(idea_row)
    if idea.claim_type not in EMPIRICAL_CLAIMS:
        return "NOT_EMPIRICAL"
    if not evidence_world_bound(state):
        return "AWAITING_ATTESTED_WORLD"
    return "UNVALIDATED"


def bridge_candidates(state: Any) -> list[str]:
    """Promising empirical ideas the evidence runtime may compile into a registered
    hypothesis — only once an attested world is bound, and each only once."""
    if not evidence_world_bound(state):
        return []
    view = discovery_view(state)
    compiled = {str(row.get("discovery_idea_id") or "") for row in _get(state, "theories", []) or []}
    return [str(row["id"]) for row in view["ideas"]
            if (view["feedback"].get(str(row["id"])) or {}).get("verdict") == PROMISING
            and idea_of(row).claim_type in EMPIRICAL_CLAIMS and str(row["id"]) not in compiled]


def stamp_generated(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the admission boundary does to every discovery record: GENERATED."""
    row = dict(payload or {})
    claimed = str(row.get("epistemic") or "GENERATED")
    if claimed != "GENERATED":
        row["promotion_refused"] = (
            f"a discovery record claimed {claimed}; discovery is GENERATED, and only the "
            "evidence runtime can admit WORLD evidence")
    row["epistemic"] = "GENERATED"
    return row


def research_promise(feedback: Mapping[str, Any] | None) -> str:
    if not feedback:
        return "UNASSESSED"
    verdict = str(feedback.get("verdict") or "")
    outcome = str((feedback.get("rollout") or {}).get("outcome") or "")
    if verdict == PROMISING:
        return "HIGH" if outcome == DEVELOPS else "MEDIUM"
    if verdict == REVISE:
        return "MEDIUM" if outcome == DEVELOPS and feedback.get("repairable", True) else "LOW"
    return "LOW"


# Critic taste, executability and discriminability are three questions.
# PROMISING / REVISE / COLLAPSE answers only the first.
RESEARCH_DIMENSIONS = (
    "importance",
    "novelty",
    "epistemic_uncertainty",
    "mechanism_quality",
    "discriminability",
    "tractability",
    "resource_cost",
    "capability_gap",
    "fatal_validity_flaw",
)
SCIENTIFIC_OBJECTS = (
    "assumption",
    "mechanism",
    "intervention",
    "measurement",
    "representation",
    "population",
    "world",
)
FAILURE_CLASSES = (
    "hypothesis_wrong",
    "assumption_wrong",
    "measurement_inadequate",
    "intervention_non_discriminating",
    "implementation_failure",
    "capability_missing",
    "resource_limitation",
    "research_question_exhausted",
)
FAILURE_CHANGES = {
    "hypothesis_wrong": "mechanism",
    "assumption_wrong": "assumption",
    "measurement_inadequate": "measurement",
    "intervention_non_discriminating": "intervention",
    "implementation_failure": "representation",
    "capability_missing": "world",
    "resource_limitation": "population",
    "research_question_exhausted": "mechanism",
}
_TASTE_CAUSES = frozenset({
    "low_novelty", "scope_mismatch", "ill_posed_problem", "incomplete_idea",
})
_DISCRIMINATION_CAUSES = frozenset({
    "unfalsifiable", "competing_explanation_unresolved",
})
_INTEGRITY_PROTOCOL_ERRORS = frozenset({
    "target_leakage",
    "unattested_or_changed_world_bytes",
    "computational_receipt_identity_mismatch",
})


def assess_research_dimensions(
    *,
    cause: str = "",
    blocking: bool = False,
    critique: Mapping[str, Any] | None = None,
    rollout: Mapping[str, Any] | None = None,
    match: Mapping[str, Any] | None = None,
    protocol_error_list: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Split taste, execution and discriminability. Not a new verdict."""
    from .requirement import (
        CAPABILITY_GAP,
        MATCH_REQUIRES_INSTRUMENTATION,
        RESOURCE_BLOCKED,
        UNSUPPORTED_WORLD,
    )

    crit = dict(critique or {})
    roll = dict(rollout or {})
    classified = dict(match or {})
    status = str(classified.get("status") or "")
    errors = {str(item) for item in (protocol_error_list or [])}
    fatal = bool(errors & _INTEGRITY_PROTOCOL_ERRORS or crit.get("fatal_validity_flaw"))
    capability = status == CAPABILITY_GAP or bool(crit.get("capability_gap"))
    cannot_run = status in {
        CAPABILITY_GAP, RESOURCE_BLOCKED, UNSUPPORTED_WORLD, MATCH_REQUIRES_INSTRUMENTATION,
    }
    cannot_discriminate = cause in _DISCRIMINATION_CAUSES
    not_publishable = cause in _TASTE_CAUSES or (blocking and cause not in _DISCRIMINATION_CAUSES
                                                 and not fatal and not capability)
    novelty = "low" if cause == "low_novelty" else "unknown"
    importance = "low" if blocking and cause in _TASTE_CAUSES else "unknown"
    mechanism = "low" if cause == "mechanism_unsupported" else "unknown"
    uncertainty = "high" if str(roll.get("outcome") or "") in {FRAGILE, COLLAPSES} else "unknown"
    return {
        "importance": importance,
        "novelty": novelty,
        "epistemic_uncertainty": uncertainty,
        "mechanism_quality": mechanism,
        "discriminability": "low" if cannot_discriminate else "unknown",
        "tractability": "low" if cannot_run else "unknown",
        "resource_cost": "high" if status == RESOURCE_BLOCKED else "unknown",
        "capability_gap": capability,
        "fatal_validity_flaw": fatal,
        "hard_execution_block": fatal,
        "not_worth_publishing": bool(not_publishable),
        "cannot_execute": cannot_run,
        "cannot_discriminate": cannot_discriminate,
        "critic_assessments": {
            "novelty": " ".join(str(crit.get("novelty_assessment") or "").split()),
            "importance": " ".join(str(crit.get("importance_assessment") or "").split()),
        },
        "basis": "derived_heuristic_not_calibrated_scientific_information_gain",
    }


def classify_research_failure(
    *,
    match: Mapping[str, Any] | None = None,
    pilot_verdict: str = "",
    execution_status: str = "",
    verdict: str = "",
    cause: str = "",
    repairable: bool = True,
) -> str:
    """What actually failed. Not which action happens to be runnable."""
    from .requirement import (
        CAPABILITY_GAP,
        MATCH_REQUIRES_INSTRUMENTATION,
        PILOT_INCONCLUSIVE,
        PILOT_NEGATIVE,
        RESOURCE_BLOCKED,
        UNSUPPORTED_WORLD,
    )

    status = str((match or {}).get("status") or "")
    if status == CAPABILITY_GAP:
        return "capability_missing"
    if status == RESOURCE_BLOCKED:
        return "resource_limitation"
    if status == UNSUPPORTED_WORLD and not repairable:
        return "research_question_exhausted"
    if str(execution_status or "") in {"failed", "blocked"}:
        return "implementation_failure"
    if pilot_verdict == PILOT_NEGATIVE:
        return "hypothesis_wrong"
    if pilot_verdict == PILOT_INCONCLUSIVE:
        return "intervention_non_discriminating"
    if cause == "unrealistic_assumption":
        return "assumption_wrong"
    if cause == "measurement_invalid" or status == MATCH_REQUIRES_INSTRUMENTATION:
        return "measurement_inadequate"
    if verdict == COLLAPSE and not repairable:
        return "research_question_exhausted"
    return ""


def scientific_object_snapshot(idea: Any) -> dict[str, str]:
    experiment = dict(getattr(idea, "experiment", {}) or {})
    uncertainties = getattr(idea, "uncertainties", ()) or ()
    return {
        "assumption": "; ".join(str(item) for item in uncertainties) or str(
            getattr(idea, "competing_explanation", "") or ""),
        "mechanism": str(getattr(idea, "mechanism", "") or ""),
        "intervention": json.dumps(experiment.get("parameters") or {}, sort_keys=True)
        if experiment.get("parameters") else str(getattr(idea, "mechanism", "") or ""),
        "measurement": str(experiment.get("metric") or getattr(idea, "validation_direction", "") or ""),
        "representation": str(experiment.get("operation") or ""),
        "population": str(getattr(idea, "research_problem", "") or ""),
        "world": str(experiment.get("operation") or getattr(idea, "claim_type", "") or ""),
    }


def redesign_changed_required_object(parent: Any, child: Any, failure_class: str) -> bool:
    needed = FAILURE_CHANGES.get(failure_class)
    if not needed:
        return False
    return scientific_object_snapshot(parent).get(needed) != scientific_object_snapshot(child).get(needed)


def operationalization_key(idea: Any) -> str:
    experiment = dict(getattr(idea, "experiment", {}) or {})
    if experiment.get("operation"):
        return digest({
            "operation": experiment.get("operation"),
            "parameters": experiment.get("parameters"),
            "metric": experiment.get("metric"),
            "expected_direction": experiment.get("expected_direction"),
        })
    return " ".join(str(getattr(idea, "optional_experiment_direction") or "").lower().split())


def same_operationalization(left: Any, right: Any) -> bool:
    first, second = operationalization_key(left), operationalization_key(right)
    return bool(first and second and first == second)


def research_program(state: Any) -> dict[str, Any]:
    """Derived research state. Heuristic bottleneck, not calibrated information gain."""
    from .requirement import capability_gaps

    view = discovery_view(state)
    contract = dict(_get(state, "contract", {}) or {})
    gaps = capability_gaps(state)
    ideas, feedback = view.get("ideas") or [], view.get("feedback") or {}
    competing = []
    for row in ideas:
        idea = idea_of(row)
        if idea.competing_explanation:
            competing.append({"idea_id": row.get("id"), "text": idea.competing_explanation})
    bottleneck = "no_research_question"
    if gaps:
        bottleneck = "capability_gap"
    elif any(str((feedback.get(str(row.get("id"))) or {}).get("verdict") or "") == REVISE
             for row in ideas):
        bottleneck = "hypothesis_or_operationalization"
    elif ideas:
        bottleneck = "evidence"
    return {
        "core_question": contract.get("root_question") or str(_get(state, "goal", "") or ""),
        "significance": contract.get("success_target") or "",
        "competing_hypotheses": competing,
        "decisive_unknowns": [row.get("reason") or row.get("missing_capability") for row in gaps]
        or [str((feedback.get(str(row.get("id"))) or {}).get("root_failure_cause") or "")
            for row in ideas if (feedback.get(str(row.get("id"))) or {}).get("root_failure_cause")
            and (feedback.get(str(row.get("id"))) or {}).get("root_failure_cause") != NO_FAILURE],
        "current_bottleneck": bottleneck,
        "capability_gaps": gaps,
        "branch_plan": [
            {"idea_id": row.get("id"), "operator": row.get("operator"), "mode": row.get("mode")}
            for row in ideas
        ],
        "kill_criteria": list(contract.get("termination_requirements") or []),
        "success_criteria": str(contract.get("success_target") or ""),
        "frontier": list(getattr(state, "research_branches", []) or []),
        "reasoning_mode": str(getattr(state, "reasoning_mode", "") or LOCAL_RESEARCH),
        "basis": "derived_heuristic_program_not_calibrated_information_gain",
    }


def _selection_key(row: Mapping[str, Any], feedback: Mapping[str, Any] | None,
                   order: int) -> tuple[Any, ...]:
    """Better research hypothesis first. Distance is deliberately absent."""
    fb = feedback or {}
    majors = sum(1 for item in fb.get("structural_validation") or []
                 if item.get("severity") == "major")
    return (PROMISE_RANK.get(research_promise(fb) if fb else "UNASSESSED", 0),
            VERDICT_RANK.get(str(fb.get("verdict") or ""), -1),
            -majors, order)


def lineages(ideas: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in ideas:
        out.setdefault(str(row.get("lineage_id") or "L1"), []).append(dict(row))
    for chain in out.values():
        chain.sort(key=lambda row: int(row.get("hop") or 1))
    return out


def discovery_view(state: Any) -> dict[str, Any]:
    ideas, feedback = discovery_records(state)
    policy = current_policy(state)
    history = []
    for row in policy_records(state):
        entry = {key: row.get(key) for key in (
            "id", "old_policy_version", "new_policy_version", "accepted", "installed",
            "reason", "proposal_digest", "proposed_policy_change", "acceptance_reasons",
            "lineage_id", "scope", "evaluation_basis", "shared_installation")}
        version = row.get("new_policy_version")
        entry["used_by_next_hop"] = next(
            (str(idea.get("id")) for idea in ideas
             if version and idea.get("policy_version") == version
             and (not row.get('lineage_id') or idea.get('lineage_id') == row['lineage_id'])), "") if row.get("installed") else ""
        history.append(entry)
    statuses = {}
    for row in ideas:
        fb = feedback.get(str(row["id"]))
        statuses[str(row["id"])] = {
            "verdict": str((fb or {}).get("verdict") or "UNVALIDATED_IDEA"),
            "research_promise": research_promise(fb),
            "evidence_status": evidence_status(state, row),
            "evidence_route": evidence_route(idea_of(row), state),
        }
    ranked = sorted(((row, index) for index, row in enumerate(ideas)),
                    key=lambda pair: _selection_key(pair[0], feedback.get(str(pair[0]["id"])),
                                                    pair[1]), reverse=True)
    selected = str(ranked[0][0]["id"]) if ranked else ""
    limits = dict(policy.get("limits") or {})
    delivered = (any(str(fb.get("verdict")) == PROMISING for fb in feedback.values())
                 or len(ideas) >= int(limits.get("max_ideas") or 8))
    return {
        "rule": "discovery-view-v1",
        "ideas": ideas,
        "feedback": feedback,
        "lineages": {lid: [str(row["id"]) for row in chain]
                     for lid, chain in lineages(ideas).items()},
        "policy": policy,
        "policy_version": str(policy.get("version") or ""),
        "policy_history": history,
        "statuses": statuses,
        "selected_idea": selected,
        "delivered": delivered,
        "remote_searches": [
            {"idea_id": row["id"], "purpose": (row.get("remote_search") or {}).get("purpose"),
             "frames": [f.get("label") for f in (row.get("remote_search") or {}).get("frames") or []]}
            for row in ideas if (row.get("remote_search") or {}).get("used")],
        "epistemic": "GENERATED",
    }


# ---------------------------------------------------------------------------
# V1 structural validation and literature grounding (deterministic)
# ---------------------------------------------------------------------------

_DISTANCE_WORDS = re.compile(r"\b(far|distant|remote|farthest|unrelated field|alien)\b", re.I)


def structural_validation(idea: CompleteIdea, *, remote_used: bool = False,
                          operator: str = "", parent: CompleteIdea | None = None,
                          ) -> list[dict[str, Any]]:
    """Is this a complete, falsifiable, internally distinct research idea?

    Diagnosis, not a gate: every finding goes into the feedback packet and
    shapes the next hop. None of them deletes the idea.
    """
    found: list[dict[str, Any]] = []

    def add(check: str, severity: str, detail: str, failure: str) -> None:
        found.append({"check": check, "severity": severity, "detail": detail,
                      "failure_mode": failure})

    for name in idea.missing():
        add("complete_idea", "major", f"{name} is empty", "incomplete_idea")
    if idea.falsifier and similarity(idea.falsifier, idea.central_claim) >= 0.9:
        add("falsifier", "major", "the falsifier restates the claim instead of naming "
            "an outcome that would refute it", "unfalsifiable")
    if idea.competing_explanation and similarity(idea.competing_explanation, idea.central_claim) >= 0.8:
        add("competing_explanation", "major", "the competing explanation restates the claim",
            "competing_explanation_unresolved")
    if idea.mechanism and similarity(idea.mechanism, idea.central_claim) >= 0.85:
        add("mechanism", "major", "the mechanism restates the claim rather than explaining it",
            "mechanism_unsupported")
    if (remote_used or operator == "TRANSFER_STRUCTURE") and not idea.structural_correspondence:
        add("structural_correspondence", "major", "a remote or transferred concept is used "
            "without a stated structural correspondence", "missing_structure")
    if idea.novelty_hypothesis and _DISTANCE_WORDS.search(idea.novelty_hypothesis):
        add("novelty_from_distance", "minor", "novelty is argued from conceptual distance; "
            "distance is not novelty", "low_novelty")
    if idea.central_claim and idea.research_problem and similarity(
            idea.central_claim, idea.research_problem) >= 0.92:
        add("copy_like", "minor", "the claim restates the problem almost verbatim",
            "low_novelty")
    if idea.mechanism and idea.optional_experiment_direction and similarity(
            idea.mechanism, idea.optional_experiment_direction) >= 0.9:
        add("known_combination", "minor", "mechanism and experiment direction are the same sentence",
            "missing_structure")
    if not idea.claim_type:
        add("claim_type", "minor", "the claim type is unstated, so it is routed as conceptual",
            "unclassified")
    if operator == REMOTE_ESCAPE and parent is not None and same_operationalization(parent, idea):
        add("invalid_escape", "major",
            "remote escape changed the metaphor but kept the same operationalization",
            "missing_structure")
    return found


def literature_grounding(state: Any, idea: CompleteIdea) -> dict[str, Any]:
    """What verified literature says about the claim — a gap is recorded, never fatal."""
    from ..livefeed import FreshWork
    from ..prior import strongest_prior
    from .ideation import _literature

    allowed = {w['cite_id'] for w in _literature(state if isinstance(state, Mapping) else state.to_dict())['works']}

    surveys = [row for row in _artifacts(state) if row.get("kind") == "literature_survey"]
    works: dict[str, Any] = {}
    for survey in surveys:
        for raw in survey.get("works") or []:
            try:
                work = FreshWork.from_dict(raw)
            except (TypeError, ValueError, KeyError):
                continue
            if work.cite_id() in allowed:
                works[work.cite_id()] = work
    transitions = sum(len(row.get("research_transitions") or []) for row in surveys)
    if not works:
        return {"status": "literature_gap", "verified_works": 0,
                "trajectory_hits": transitions,
                "detail": ("no verified literature was retrieved; the idea stands, and its "
                           "novelty and grounding remain unchecked"),
                "required": "a verified literature check of the central claim"}
    prior = strongest_prior(idea.central_claim, list(works.values()))
    cited = sorted(set(idea.grounding) & set(works))
    if prior is not None and getattr(prior, "kills", False):
        return {"status": "prior_states_claim", "verified_works": len(works),
                "trajectory_hits": transitions, "prior": prior.to_dict(), "cited": cited,
                "detail": "verified literature already states this claim"}
    return {"status": "grounded" if cited else "checked_not_cited",
            "verified_works": len(works), "trajectory_hits": transitions, "cited": cited,
            "prior": prior.to_dict() if prior is not None else None}


# ---------------------------------------------------------------------------
# Operator selection and lineage stagnation
# ---------------------------------------------------------------------------


def normalize_failure(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not text or text in {"none", "no_failure", "n/a", "null"}:
        return NO_FAILURE
    if text in FAILURE_MODES:
        return text
    for needle, mode in _FAILURE_SYNONYMS:
        if needle in text:
            return mode
    return "unclassified"


def _pairs_for(chain: Sequence[Mapping[str, Any]],
               feedback: Mapping[str, Mapping[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(dict(row), dict(feedback.get(str(row.get("id") or "")) or {})) for row in chain]


def select_operator(policy: Mapping[str, Any], failure: str,
                    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]] = ()) -> tuple[str, str]:
    """P(operator | failure mode) from the installed policy, minus what this lineage spent.

    An operator is spent on a failure when a hop applied it to that failure and
    its own feedback came back with the same failure: repeating it would be a
    re-read of a known answer.
    """
    mapping = dict(policy.get("operator_for_failure") or {})
    ranked = list(mapping.get(failure) or mapping.get("unclassified") or ["REFORMULATE"])
    spent: set[str] = set()
    for index in range(1, len(pairs)):
        parent_fb = pairs[index - 1][1]
        child, child_fb = pairs[index]
        if (normalize_failure(parent_fb.get("root_failure_cause")) == failure
                and normalize_failure(child_fb.get("root_failure_cause")) == failure):
            spent.add(str(child.get("operator") or ""))
    version = str(policy.get("version") or "")
    for op in ranked:
        if op not in spent:
            return op, f"{version}:{failure}->{op}"
    return ranked[0], f"{version}:{failure}->{ranked[0]}:every_ranked_operator_already_spent"


def lineage_stagnation(policy: Mapping[str, Any],
                       pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> list[str]:
    """Why a lineage is in a local optimum, or [] when it is still moving."""
    rules = dict(policy.get("stagnation") or {})
    reasons: list[str] = []
    judged = [(idea, fb) for idea, fb in pairs if fb]
    streak = int(rules.get("same_failure_streak") or 3)
    tail = judged[-streak:]
    if len(tail) >= streak and all(fb.get("verdict") != PROMISING for _, fb in tail):
        causes = {normalize_failure(fb.get("root_failure_cause")) for _, fb in tail}
        if len(causes) == 1 and NO_FAILURE not in causes:
            reasons.append(f"same_root_failure_{causes.pop()}_x{streak}")
    similar = int(rules.get("similar_streak") or 2)
    threshold = float(rules.get("structure_similarity") or 0.85)
    ideas = [idea_of(idea) for idea, _ in pairs]
    if len(ideas) > similar:
        recent = ideas[-(similar + 1):]
        if all(similarity(f"{a.central_claim} {a.mechanism}", f"{b.central_claim} {b.mechanism}") >= threshold
               for a, b in zip(recent, recent[1:])):
            reasons.append("core_structure_unchanged")
    if judged:
        last = judged[-1][1]
        if last.get("verdict") == COLLAPSE and not last.get("repairable", True):
            reasons.append("lineage_judged_dead_end")
    return reasons


# ---------------------------------------------------------------------------
# Feedback composition (deterministic over V1, grounding, rollout and critique)
# ---------------------------------------------------------------------------


def _strings(value: Any, limit: int = 12) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in value if isinstance(value, (list, tuple)) else [value]:
        if isinstance(item, Mapping):
            item = "; ".join(f"{k}: {v}" for k, v in item.items())
        text = " ".join(str(item or "").split())
        if text and text not in out:
            out.append(text)
    return out[:limit]


def normalize_rollout(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """A ResearchWorldRollout. Forward research reasoning — GENERATED, never evidence."""
    row = dict(raw or {})
    outcome = str(row.get("outcome") or "").strip().upper()
    normalized_from = ""
    if outcome not in OUTCOMES:
        normalized_from, outcome = outcome or "missing", FRAGILE
    rollout = {
        "likely_next_questions": _strings(row.get("likely_next_questions")),
        "likely_competing_explanations": _strings(row.get("likely_competing_explanations")),
        "likely_missing_knowledge": _strings(row.get("likely_missing_knowledge")),
        "predicted_contradictions": _strings(row.get("predicted_contradictions")),
        "required_evidence": _strings(row.get("required_evidence")),
        "likely_failure_point": " ".join(str(row.get("likely_failure_point") or "").split()),
        "what_would_kill_the_idea": " ".join(str(row.get("what_would_kill_the_idea") or "").split()),
        "what_result_would_strengthen_it": " ".join(
            str(row.get("what_result_would_strengthen_it") or "").split()),
        "possible_research_path": _strings(row.get("possible_research_path")),
        "dependency_structure": _strings(row.get("dependency_structure")),
        "outcome": outcome,
        "reason": " ".join(str(row.get("reason") or "").split()) or "no reason given",
        "epistemic": "GENERATED",
        "basis": "research_world_simulation_not_evidence",
    }
    if normalized_from:
        rollout["outcome_normalized_from"] = normalized_from
    return rollout


def compose_feedback(
    *,
    idea_row: Mapping[str, Any],
    structural: Sequence[Mapping[str, Any]],
    grounding: Mapping[str, Any],
    rollout: Mapping[str, Any],
    critique: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    state: Any,
) -> dict[str, Any]:
    """A ResearchFeedbackPacket. The verdict is derived here, not asserted by a model.

    PROMISING needs a structurally complete idea, a literature check that did
    not find the claim already stated, a simulation that does not predict
    collapse and a critic who found nothing blocking. It never needs evidence
    an early idea cannot have yet: replication, experiments and baselines go
    into `required_evidence`, not into the verdict.
    """
    crit = dict(critique or {})
    idea = idea_of(idea_row)
    from .experiment import adaptive_criterion_sources
    adaptive_sources = adaptive_criterion_sources(idea.experiment, state)
    structural = list(structural)
    if adaptive_sources:
        observed_digests = {r.get('world_digest') for r in _get(state, 'evidence_records', []) or []}
        fresh_bound = any(w.get('id') == _get(state, 'world_id', '')
            and w.get('research_role') == 'heldout' and w.get('attested')
            and w.get('digest') and w['digest'] not in observed_digests
            for w in _get(state, 'world_versions', []) or [])
        structural.append({'check': 'posthoc_criterion_change', 'severity': 'minor' if fresh_bound else 'major',
            'failure_mode': 'measurement_invalid',
            'detail': 'Decision criteria changed after observing ' + ', '.join(
                str(r.get('evidence_id')) for r in adaptive_sources) +
                '. This is an exploratory revision: preserve the original verdict; '
                'register the revised question and evaluate on genuinely unseen data.'})
    majors = [row for row in structural if row.get("severity") == "major"]
    repairable = crit.get("repairable")
    repairable = True if repairable is None else bool(repairable)
    blocking = bool(crit.get("blocking"))
    critic_cause = normalize_failure(crit.get("root_failure_cause"))
    if majors:
        cause = normalize_failure(majors[0].get("failure_mode"))
    elif grounding.get("status") == "prior_states_claim":
        cause = "low_novelty"
    elif critic_cause != NO_FAILURE:
        cause = critic_cause
    elif blocking:
        cause = "unclassified"
    else:
        cause = NO_FAILURE
    outcome = str(rollout.get("outcome") or FRAGILE)
    if outcome == COLLAPSES and not repairable:
        # A generated forecast is an objection, not an observed dead end.
        # Preserve the raw judgement below, but permit bounded investigation.
        verdict = REVISE
        repairable = True
    elif not majors and (cause == NO_FAILURE or (cause == 'low_novelty' and not blocking)) and outcome != COLLAPSES:
        verdict = PROMISING
    else:
        verdict = REVISE
    if verdict == COLLAPSE and cause == NO_FAILURE:
        cause = critic_cause if critic_cause != NO_FAILURE else "unclassified"
    limits = dict(policy.get("limits") or {})
    if verdict == PROMISING:
        operator = BRANCH_DIVERSIFICATION if limits.get("branch_after_promising") else ""
        basis = "promising_ideas_branch_rather_than_stop" if operator else "promising"
    elif verdict == COLLAPSE and not repairable:
        operator, basis = REMOTE_ESCAPE, "lineage_judged_dead_end"
    else:
        operator, basis = select_operator(policy, cause, list(pairs) + [(dict(idea_row), {
            "root_failure_cause": cause})])
    route = evidence_route(idea, state)
    match: dict[str, Any] = {}
    protocol_error_list: list[str] = []
    try:
        from .experiment import protocol_errors
        from .requirement import classify_world_match, compile_experiment_requirement

        req = compile_experiment_requirement(
            idea, idea_id=str(idea_row.get("id") or ""), feedback=crit)
        match = classify_world_match(req)
        protocol_error_list = protocol_errors(req)
    except (TypeError, ValueError, KeyError):
        match, protocol_error_list = {}, []
    dimensions = assess_research_dimensions(
        cause=cause, blocking=blocking, critique=crit, rollout=rollout,
        match=match, protocol_error_list=protocol_error_list)
    failure_class = classify_research_failure(
        match=match, verdict=verdict, cause=cause, repairable=repairable)
    required = _strings(crit.get("required_evidence")) + [
        item for item in _strings(rollout.get("required_evidence"))
        if item not in _strings(crit.get("required_evidence"))]
    if grounding.get("status") == "literature_gap":
        required.append(str(grounding.get("required") or "a verified literature check"))
    if route == "V4_EVIDENCE_RUNTIME":
        required.append("a pre-registered two-arm test on the attested world, judged by the "
                        "validity contract (placebo worlds, oracle, sample, ablation) and replicated")
    elif route == "AWAITING_ATTESTED_WORLD":
        required.append("an attested frozen world on which the empirical claim can be tested")
    structural_score = round(max(0.0, 1.0 - 0.25 * len(majors)
                                 - 0.05 * (len(structural) - len(majors))), 4)
    from .inquiry import investment_packet
    investment = investment_packet(idea.to_dict(), crit, grounding, state)
    return {
        "kind": FEEDBACK_KIND,
        "id": f"fb-{idea_row.get('id')}",
        "idea_id": str(idea_row.get("id") or ""),
        "lineage_id": str(idea_row.get("lineage_id") or ""),
        "hop": int(idea_row.get("hop") or 1),
        "verdict": verdict,
        "research_promise": research_promise({"verdict": verdict, "rollout": rollout,
                                              "repairable": repairable}),
        "strongest_positive_signal": " ".join(str(crit.get("strongest_positive_signal") or "").split()),
        "strongest_negative_signal": " ".join(str(crit.get("strongest_negative_signal") or "").split())
        or (majors[0]["detail"] if majors else ""),
        "what_survives": _strings(crit.get("what_survives")),
        "what_failed": [row["detail"] for row in majors] + _strings(crit.get("what_failed")),
        "root_failure_cause": cause,
        "critic_root_failure_cause": critic_cause,
        "strongest_competing_explanation": " ".join(
            str(crit.get("strongest_competing_explanation") or idea.competing_explanation).split()),
        "repairable": repairable,
        "critic_repairable": crit.get("repairable"),
        "investment": investment,
        "blocking": blocking,
        "recommended_next_operator": operator,
        "operator_basis": basis,
        "critic_suggested_operator": str(crit.get("suggested_operator") or "").upper(),
        "preserve_constraints": _strings(crit.get("preserve_constraints")),
        "change_constraints": _strings(crit.get("change_constraints")),
        "mistakes_to_avoid": _strings(crit.get("mistakes_to_avoid")),
        "new_questions": _strings(crit.get("new_questions")) or _strings(
            rollout.get("likely_next_questions")),
        "required_evidence": required,
        "structural_validation": [dict(row) for row in structural],
        "literature_grounding": dict(grounding),
        "rollout": dict(rollout),
        "evidence_route": route,
        # Kept apart on purpose: none of these is a posterior over the others.
        "research_dimensions": dimensions,
        "failure_class": failure_class,
        "must_change": FAILURE_CHANGES.get(failure_class, ""),
        "world_match": match,
        "assessments": {
            "model_assessment": {
                "novelty": " ".join(str(crit.get("novelty_assessment") or "").split()),
                "importance": " ".join(str(crit.get("importance_assessment") or "").split()),
                "basis": "GENERATED critic judgement",
            },
            "heuristic_assessment": {"structural_score": structural_score,
                                     "basis": "deterministic structural checks"},
            "structural_validation": {"major": len(majors), "minor": len(structural) - len(majors)},
            "literature_grounding": str(grounding.get("status") or ""),
            "human_assessment": None,
            "empirical_evidence": evidence_status(state, idea_row),
            "research_dimensions": dimensions,
        },
        "policy_version": str(policy.get("version") or ""),
        "epistemic": "GENERATED",
    }


# ---------------------------------------------------------------------------
# Planning the next discovery moves
# ---------------------------------------------------------------------------

MOVE_PRIORITY = {
    CRITIQUE: 5.0,
    EVOLVE_POLICY: 4.0,
    REMOTE_BOOTSTRAP: 3.6,
    EVOLVE: 3.4,
    REMOTE_ESCAPE: 3.2,
    BRANCH_DIVERSIFICATION: 2.0,
}


def move_priority(action_type: str, extra: Mapping[str, Any]) -> float:
    """Fixed and explainable: validate what exists before making more of it."""
    if action_type == IDEATE:
        return MOVE_PRIORITY.get(str(extra.get("mode") or ""), 1.0)
    return MOVE_PRIORITY.get(action_type, 1.0)


def _move(action: str, target: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"action": action, "target": target, "reason": reason,
            "extra": {key: value for key, value in extra.items() if value not in (None, "")}}


def _settled_but_unanswered(state: Any, idea_id: str) -> bool:
    """Did this lineage confirm something the mission still cannot use?

    Reads the derived answer rather than re-deriving one: the question is
    unresolved, and this idea nevertheless has a corroborating confirmatory or
    replication reading. Polishing it further cannot change either fact.
    """
    from ..validity import corroborates

    answer = dict((getattr(state, "quality", {}) or {}).get("question_answer") or {})
    if answer.get("status") != "unresolved":
        return False
    return any(
        str(row.get("discovery_idea_id") or "") == str(idea_id)
        and str(row.get("study_stage") or "") in {"confirmatory", "replication"}
        and not row.get("invalidated") and corroborates(row)
        for row in getattr(state, "evidence_records", []) or ())


def registration_gap_for(state: Any, idea_id: str) -> dict[str, Any]:
    """The recorded reason this idea's registered experiment cannot run, if any.

    Read from the frontier projection rather than recompiled here: one place
    decides whether a registration is runnable, and everything else reads it.
    """
    if not idea_id:
        return {}
    gaps = (getattr(state, "frontier", {}) or {}).get("measurement_gaps") or ()
    return next((dict(row) for row in gaps
                 if str(row.get("idea_id") or "") == str(idea_id)
                 and row.get("kind") == "registration_gap"), {})


def plan_moves(state: Any) -> list[dict[str, Any]]:
    """What discovery could do next, from its own records and the installed policy.

    Validate every idea before evolving it. A lineage whose head needs revision
    evolves by the operator its feedback calls for, without a latent jump. Only
    a stuck or dead lineage escapes remotely. A promising idea stays a
    candidate and may seed a different lineage; it does not end exploration.
    """
    if not root_question(state):
        return []
    view = discovery_view(state)
    policy = view["policy"]
    version = view["policy_version"]
    limits = dict(policy.get("limits") or {})
    remote = dict(policy.get("remote") or {})
    ideas, feedback = view["ideas"], view["feedback"]
    moves: list[dict[str, Any]] = []
    from .exploratory import environment, records, attempts, proposal_error
    exploratory = {r['id'] for r in ideas if environment(state)
                   and idea_of(r).exploration and not proposal_error(idea_of(r).exploration)}
    heads = {chain[-1]['id'] for chain in lineages(ideas).values()}
    from .dependencies import dependency_view
    dependencies = dependency_view(state)
    from .inquiry import knowledge_move, inquiry_request
    for row in ideas:
        if (idea_of(row).exploration or idea_of(row).knowledge_request) and row['id'] not in heads:
            continue  # a child decision supersedes its parent's unexecuted plan
        if idea_of(row).knowledge_request:
            inquiry = knowledge_move(state, row, feedback.get(row['id'], {}))
            if inquiry:
                moves.append(inquiry)
                continue
            if inquiry_request(row, feedback.get(row['id'], {})):
                continue  # interpret the returned sources or budget gap before executing code
        if row['id'] in exploratory:
            if not records(state, row['id']) and not attempts(state, row['id']) and dependencies.get(row['id'], {}).get('status') == 'current':
                moves.append(_move('EXPLORE', row['id'], 'execute_agent_designed_exploration',
                                   discovery_idea_id=row['id'], lineage_id=row.get('lineage_id')))
            continue
        if str(row["id"]) not in feedback:
            moves.append(_move(CRITIQUE, str(row["id"]), "validate_the_complete_idea",
                               idea_id=str(row["id"]), lineage_id=row.get("lineage_id"),
                               hop=int(row.get("hop") or 1)))
    if not ideas:
        moves.append(_move(IDEATE, "intent", "leave_the_local_basin_once", mode=REMOTE_BOOTSTRAP,
                           lineage_id="L1", hop=1, remote=True, policy_version=version))
        return moves
    if len(ideas) < int(limits.get("max_ideas") or 8):
        chains = lineages(ideas)
        for lid, chain in chains.items():
            policy = current_policy(state, lid)
            version = str(policy.get('version') or '')
            limits = dict(policy.get('limits') or {})
            remote = dict(policy.get('remote') or {})
            head = chain[-1]
            fb = feedback.get(str(head["id"]))
            if idea_of(head).knowledge_request:
                from .inquiry import inquiry_request, inquiry_progress
                request = inquiry_request(head, fb or {})
                if request:
                    progress = inquiry_progress(state, request)
                    if progress['status'] == 'ready':
                        continue
                    moves.append(_move(IDEATE, head['id'], 'interpret_targeted_knowledge_or_budget_gap',
                        mode=EVOLVE, operator='', parent_idea_id=head['id'], lineage_id=lid,
                        question_id=str(head.get('question_id') or ''),
                        decision_request='interpret_targeted_knowledge_or_budget_gap',
                        hop=int(head.get('hop') or 1) + 1, remote=False, policy_version=version))
                    continue
            if head['id'] in exploratory:
                if records(state, head['id']) or attempts(state, head['id']) or dependencies.get(head['id'], {}).get('status') == 'reconsideration_required':
                    moves.append(_move(IDEATE, head['id'], 'respond_to_exploratory_observation',
                        mode=EVOLVE, operator='', parent_idea_id=head['id'], lineage_id=lid,
                        question_id=str(head.get('question_id') or ''),
                        decision_request='interpret_unvalidated_execution_and_choose_next_unknown',
                        hop=int(head.get('hop') or 1) + 1, remote=False, policy_version=version))
                continue
            # An empirical observation changes the scientific situation before
            # any critic/operator policy is consulted. Reproduction alone does
            # not identify a repair, and a refutation does not prescribe one.
            live = _live_evidence(state, str(head['id']))
            observed = [e for e in live if e.get('epistemic') == 'WORLD' and e.get('attested')]
            negative = [e for e in observed if e.get('outcome') == 'weakens']
            ambiguous = [e for e in observed if e.get('outcome') == 'uninformative']
            insufficient = [e for e in observed if any(c.get('check') == 'sample'
                and c.get('status') in {'fail', 'undefined'} for c in (e.get('validity') or {}).get('checks', []))]
            if negative or ambiguous or insufficient:
                if not has_failure_diagnosis(state, str(head['id'])):
                    continue
                if (negative and not insufficient and not any(e.get('study_stage') in {'confirmatory', 'replication'} for e in negative)
                        and any(e.get('study_stage') in {'pilot', 'scientific_pilot'} for e in negative)):
                    continue  # settle the registered prediction before replacing it
                beliefs = getattr(state, 'belief_states', {}) or {}
                refuted = any(beliefs.get(e.get('theory_id'), {}).get('status') == 'refuted'
                              and e.get('evidence_id') in beliefs.get(e.get('theory_id'), {}).get('contradicting_evidence', [])
                              for e in negative)
                moves.append(_move(IDEATE, str(head['id']), 'evidence_requires_research_decision',
                    mode=EVOLVE, operator='', operator_basis='valid_observations_not_failure_mapping',
                    parent_idea_id=str(head['id']), lineage_id=lid,
                    question_id=str(head.get('question_id') or ''),
                    decision_request=('remaining_question_after_scoped_refutation' if refuted
                                      else 'measurement_or_identification_unresolved'),
                    evidence_ids=list(dict.fromkeys(e.get('evidence_id') for e in negative + ambiguous + insufficient)),
                    hop=int(head.get('hop') or 1) + 1, remote=False, policy_version=version))
                continue
            from .judgment import revision_request
            sent_back = revision_request(state)
            if sent_back and str(head["id"]) in sent_back["idea_ids"]:
                # Placed before the feedback check below on purpose. A registered
                # local experiment goes to its pilot without a critic verdict, so
                # a lineage can be measured, confirmed and judged while `fb` is
                # still empty — and the first live panel's verdict was skipped
                # here for exactly that reason.
                #
                # The measurement held and the panel still found the result
                # known, or not worth knowing. That is a verdict on the idea, not
                # on the experiment, so the idea goes back — told what the panel
                # said would have to change. A known result is sent outward for
                # a frame the local literature does not already hold.
                moves.append(_move(
                    IDEATE, str(head["id"]), "judged_result_falls_short_of_the_contract",
                    mode=sent_back["mode"], operator=sent_back["operator"],
                    operator_basis=f"judgment_panel:{sent_back['dimension']}:{sent_back['judged']}",
                    parent_idea_id=str(head["id"]), lineage_id=lid,
                    question_id=str(head.get("question_id") or ""),
                    hop=int(head.get("hop") or 1) + 1,
                    addresses_feedback=str((fb or {}).get("id") or ""),
                    must_change=sent_back["must_change"],
                    judged_below_the_bar=True, addresses_review=sent_back["review_id"],
                    remote=sent_back["dimension"] == "novelty", policy_version=version))
                continue
            if not fb:
                continue
            from .inquiry import knowledge_move
            inquiry = knowledge_move(state, head, fb)
            if inquiry is not None:
                moves.append(inquiry)
                # Keep revision runnable: budget/admission may defer the survey.
            gap = registration_gap_for(state, str(head["id"]))
            if gap:
                # The head's registered experiment cannot run. No verdict makes
                # that better, and no design revision can rewrite a frozen
                # protocol — a live run redesigned twice and changed nothing,
                # then proposed three unrelated ideas that were refused for the
                # same reason. The one move that repairs a registration is a new
                # proposal told exactly what has to change.
                moves.append(_move(
                    IDEATE, str(head["id"]), "registered_experiment_cannot_run_as_written",
                    mode=EVOLVE, operator="MEASUREMENT_REPAIR",
                    operator_basis="registration_gap:" + str(gap.get("gap") or ""),
                    parent_idea_id=str(head["id"]), lineage_id=lid,
                    question_id=str(head.get("question_id") or ""),
                    hop=int(head.get("hop") or 1) + 1,
                    addresses_feedback=str(fb.get("id") or ""),
                    must_change=str(gap.get("gap") or "registration"),
                    registration_gap=dict(gap), remote=False, policy_version=version))
                continue
            if _settled_but_unanswered(state, str(head["id"])):
                # The experiment worked and the question is still open. Every
                # existing trigger here reads a *failure* — negative, ambiguous,
                # insufficient, blocked, killed, stalled — so a lineage that
                # succeeded and still did not answer the mission had nothing to
                # do but polish itself. Four live missions spent twenty-six
                # ticks each on REVIEW / REDESIGN / THEORIZE over one idea for
                # exactly this reason. A confirmed result that cannot close the
                # question is a reason to look somewhere else, not a reason to
                # keep sanding the same one.
                moves.append(_move(
                    IDEATE, str(head["id"]), "confirmed_result_leaves_the_question_open",
                    mode=EVOLVE, operator="SCOPE_SHIFT",
                    operator_basis="settled_operational_result_without_a_mission_answer",
                    parent_idea_id=str(head["id"]), lineage_id=lid,
                    question_id=str(head.get("question_id") or ""),
                    hop=int(head.get("hop") or 1) + 1,
                    addresses_feedback=str(fb.get("id") or ""),
                    must_change="scope",
                    remote=False, policy_version=version))
                continue
            if fb.get("verdict") == PROMISING:
                from .requirement import (
                    CAPABILITY_GAP,
                    PILOT_INCONCLUSIVE,
                    PILOT_NEGATIVE,
                    RESOURCE_BLOCKED,
                    UNSUPPORTED_WORLD,
                    execution_feedback,
                    match_of_idea,
                    requirement_of_idea,
                )

                match = match_of_idea(state, head)
                pilots = [item for item in getattr(state, "evidence_records", []) or []
                          if item.get("discovery_idea_id") == str(head["id"])
                          and item.get("study_stage") in {"pilot", "calibration", "scientific_pilot"}]
                # A missing executor is a capability gap, not the end of discovery.
                blocked = match["status"] in {
                    RESOURCE_BLOCKED, UNSUPPORTED_WORLD, CAPABILITY_GAP}
                killed = any(item.get("pilot_verdict") == PILOT_NEGATIVE for item in pilots)
                stalled = (len(pilots) >= 2
                           and all(item.get("pilot_verdict") == PILOT_INCONCLUSIVE for item in pilots))
                if not (blocked or killed or stalled):
                    continue
                if (killed or stalled) and not has_failure_diagnosis(state, str(head["id"])):
                    continue
                req = requirement_of_idea(state, head)
                verdict = (PILOT_NEGATIVE if killed
                           else PILOT_INCONCLUSIVE if stalled
                           else str(match.get("status") or ""))
                signal = execution_feedback(req, match, pilot_verdict=verdict if killed or stalled else "")
                if not signal.get("repairable") and not killed:
                    continue
                failure = str(signal.get("failure_type") or "scope_mismatch")
                operator = str(signal.get("suggested_operator") or "SCOPE_SHIFT")
                moves.append(_move(
                    IDEATE, str(head["id"]), f"evolve_by_execution:{failure}",
                    mode=EVOLVE, operator='' if blocked else operator,
                    decision_request='question_feasibility_unresolved' if blocked else '',
                    operator_basis="execution_failure:" + failure,
                    parent_idea_id=str(head["id"]), lineage_id=lid,
                    hop=int(head.get("hop") or 1) + 1,
                    addresses_feedback=str(fb.get("id") or ""),
                    failure_class=signal.get("failure_class") or "",
                    must_change='' if blocked else signal.get("must_change") or "",
                    remote=False, policy_version=version))
                continue
            if len(chain) >= int(limits.get("max_hops_per_lineage") or 4):
                continue
            pairs = _pairs_for(chain, feedback)
            reasons = lineage_stagnation(policy, pairs)
            escapes = sum(1 for row in chain if row.get("mode") == REMOTE_ESCAPE)
            may_escape = (bool(remote.get("escape_on_stagnation", True))
                          and escapes < int(remote.get("max_escapes_per_lineage") or 0))
            failure = normalize_failure(fb.get("root_failure_cause"))
            if reasons and may_escape:
                mode, operator, basis = REMOTE_ESCAPE, REMOTE_ESCAPE, "stagnation:" + ",".join(reasons)
            elif fb.get("verdict") == COLLAPSE and not fb.get("repairable", True):
                if not may_escape:
                    continue  # a dead lineage with its escape spent is closed, and archived
                mode, operator, basis = REMOTE_ESCAPE, REMOTE_ESCAPE, "lineage_judged_dead_end"
            else:
                operator, basis = select_operator(policy, failure, pairs)
                if operator == REMOTE_ESCAPE and not may_escape:
                    ranked = [op for op in (policy.get("operator_for_failure") or {}).get(failure) or []
                              if op not in {REMOTE_ESCAPE, BRANCH_DIVERSIFICATION}]
                    operator = ranked[0] if ranked else "REFORMULATE"
                    basis += ":escape_spent"
                mode = REMOTE_ESCAPE if operator == REMOTE_ESCAPE else EVOLVE
            moves.append(_move(
                IDEATE, str(head["id"]), f"evolve_by_feedback:{failure}", mode=mode,
                operator=operator, operator_basis=basis, parent_idea_id=str(head["id"]),
                lineage_id=lid, hop=int(head.get("hop") or 1) + 1,
                addresses_feedback=str(fb.get("id") or ""), remote=mode == REMOTE_ESCAPE,
                failure_class=fb.get("failure_class") or "",
                must_change=fb.get("must_change") or "",
                stagnation=reasons or None, policy_version=version))
        # A new lineage starts from the frozen default; a sibling's learned
        # policy cannot control its limits, remote mode or version.
        branch_policy = current_policy(state, f"L{len(chains) + 1}")
        branch_limits = branch_policy.get('limits') or {}
        if branch_limits.get("branch_after_promising") and len(chains) < int(branch_limits.get("max_lineages") or 3):
            branched = {str(row.get("parent_id") or "") for row in ideas
                        if row.get("mode") == BRANCH_DIVERSIFICATION}
            for row in ideas:
                fb = feedback.get(str(row["id"])) or {}
                if fb.get("verdict") != PROMISING or str(row["id"]) in branched:
                    continue
                if _live_evidence(state, str(row['id'])):
                    continue  # actual evidence, not reviewer approval, governs continuation
                kind = REMOTE_BRANCH if (branch_policy.get('remote') or {}).get("branch_remote_first", True) else "MECHANISM_REPAIR"
                moves.append(_move(
                    IDEATE, str(row["id"]), "explore_an_alternative_lineage",
                    mode=BRANCH_DIVERSIFICATION, operator=BRANCH_DIVERSIFICATION,
                    operator_basis="promising_idea_seeds_a_different_lineage", branch_kind=kind,
                    parent_idea_id=str(row["id"]), lineage_id=f"L{len(chains) + 1}", hop=1,
                    addresses_feedback=str(fb.get("id") or ""), remote=kind == REMOTE_BRANCH,
                    policy_version=branch_policy['version']))
                break
    for lid in sorted(lineages(ideas)):
        proposal = propose_policy_update(state, lineage_id=lid)
        if proposal is None:
            continue
        recorded = {str(row.get("proposal_digest") or "") for row in view["policy_history"]}
        if proposal["proposal_digest"] not in recorded:
            moves.append(_move(EVOLVE_POLICY, version, "observed_operator_outcomes_disagree_with_policy",
                               proposal_digest=proposal["proposal_digest"], lineage_id=lid))
    return moves


# ---------------------------------------------------------------------------
# Evolving the exploration policy
# ---------------------------------------------------------------------------


def _improved(parent_fb: Mapping[str, Any], child_fb: Mapping[str, Any]) -> bool:
    """Did applying an operator to a failure leave a better idea behind?"""
    before = VERDICT_RANK.get(str(parent_fb.get("verdict") or ""), 0)
    after = VERDICT_RANK.get(str(child_fb.get("verdict") or ""), 0)
    if after != before:
        return after > before
    return (normalize_failure(child_fb.get("root_failure_cause"))
            != normalize_failure(parent_fb.get("root_failure_cause")))


def operator_outcomes(state: Any, lineage_id: str = '') -> dict[tuple[str, str], dict[str, int]]:
    """Every (failure, operator) the archive has tried, and how often it helped."""
    ideas, feedback = discovery_records(state, include_retired=True)
    by_id = {str(row["id"]): row for row in ideas}
    stats: dict[tuple[str, str], dict[str, int]] = {}
    for row in ideas:
        if lineage_id and str(row.get('lineage_id') or 'L1') != lineage_id:
            continue
        if int(row.get('project_epoch') or 0) != int(_get(state, 'project_epoch', 0) or 0):
            continue
        parent = by_id.get(str(row.get("parent_id") or ""))
        if parent and int(parent.get('project_epoch') or 0) != int(_get(state, 'project_epoch', 0) or 0):
            continue
        child_fb = feedback.get(str(row["id"]))
        parent_fb = feedback.get(str(parent["id"])) if parent else None
        if not (parent and child_fb and parent_fb) or row.get("mode") == BRANCH_DIVERSIFICATION:
            continue
        key = (normalize_failure(parent_fb.get("root_failure_cause")), str(row.get("operator") or ""))
        bucket = stats.setdefault(key, {"attempts": 0, "improvements": 0})
        bucket["attempts"] += 1
        bucket["improvements"] += int(_improved(parent_fb, child_fb))
    return stats


def propose_policy_update(state: Any, lineage_id: str = '') -> dict[str, Any] | None:
    """A policy change the archive supports, or None.

    Demote the top operator for a failure mode once it has been applied twice
    to that failure without leaving a better idea, in favour of the next ranked
    operator the archive has not seen fail. Roll the last update back when the
    operator it promoted has since failed twice.
    """
    lineage_id = policy_lineage(state, lineage_id)
    if not lineage_id:
        return None
    policy = current_policy(state, lineage_id)
    stats = operator_outcomes(state, lineage_id)
    records = scoped_policy_records(state, lineage_id)
    mapping = dict(policy.get("operator_for_failure") or {})
    changes: list[dict[str, Any]] = []
    patterns: list[dict[str, Any]] = []
    for failure, ranked in sorted(mapping.items()):
        ranked = list(ranked)
        if len(ranked) < 2:
            continue
        top = ranked[0]
        seen = stats.get((failure, top), {"attempts": 0, "improvements": 0})
        if seen["attempts"] < 2 or seen["improvements"]:
            continue
        patterns.append({"failure_mode": failure, "operator": top, **seen})
        alternative = next((op for op in ranked[1:]
                            if stats.get((failure, op), {}).get("improvements")
                            or not stats.get((failure, op), {}).get("attempts")), None)
        if alternative is None:
            continue
        changes.append({"kind": "demote", "failure_mode": failure, "demoted": top,
                        "promoted": alternative,
                        "ranked_before": ranked,
                        "ranked_after": [alternative] + [op for op in ranked if op != alternative]})
    parent = str(policy.get("parent_version") or "")
    if not changes and parent:
        for row in reversed(records):
            if row.get("installed") and row.get("new_policy_version") == policy.get("version"):
                for change in row.get("proposed_policy_change") or []:
                    promoted = str(change.get("promoted") or "")
                    seen = stats.get((str(change.get("failure_mode") or ""), promoted),
                                     {"attempts": 0, "improvements": 0})
                    if seen["attempts"] >= 2 and not seen["improvements"]:
                        patterns.append({"failure_mode": change.get("failure_mode"),
                                         "operator": promoted, **seen})
                        changes.append({"kind": "rollback", "to_version": parent,
                                        "failure_mode": change.get("failure_mode"),
                                        "demoted": promoted})
                break
    if not changes:
        return None
    count = len([row for row in records if row.get("installed")])
    candidate = copy.deepcopy(policy)
    if any(change["kind"] == "rollback" for change in changes):
        target = next((row.get("policy") for row in records
                       if row.get("new_policy_version") == parent), None)
        candidate = copy.deepcopy(dict(target)) if isinstance(target, Mapping) else copy.deepcopy(DEFAULT_POLICY)
    else:
        for change in changes:
            candidate["operator_for_failure"][change["failure_mode"]] = change["ranked_after"]
    candidate["version"] = f"exploration-policy-v{count + 2}"
    candidate["parent_version"] = str(policy.get("version") or "")
    body = {"old": policy.get("version"), "changes": changes, "candidate": candidate,
            "lineage_id": lineage_id}
    return {
        "lineage_id": lineage_id,
        "old_policy_version": str(policy.get("version") or ""),
        "observed_failure_patterns": patterns,
        "proposed_policy_change": changes,
        "reason": "; ".join(
            f"{c['failure_mode']}: {c.get('demoted')} applied {p['attempts']}x without improvement"
            for c, p in zip(changes, patterns)),
        "candidate": candidate,
        "proposal_digest": digest(body),
    }


def evaluate_policy_update(state: Any, proposal: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Scope, shape, then replay against the archive. Deterministic and auditable."""
    candidate = dict(proposal.get("candidate") or {})
    ok, reasons = validate_policy(candidate)
    stats = operator_outcomes(state, str(proposal.get('lineage_id') or ''))
    for change in proposal.get("proposed_policy_change") or []:
        if change.get("kind") != "demote":
            continue
        promoted = stats.get((change["failure_mode"], change["promoted"]), {})
        if promoted.get("attempts") and not promoted.get("improvements"):
            ok = False
            reasons.append(f"{change['promoted']} has also failed on {change['failure_mode']}")
    return ok, reasons


def execute_evolve_policy(env: Any, action: Any) -> Any:
    """EVOLVE_POLICY: install a better exploration policy, or record why not.

    The policy only chooses discovery moves. The admission boundary refuses any
    event but an `exploration_policy` artifact from this action, and the policy
    schema refuses any key naming a kernel surface.
    """
    from .events import make_event
    from .kernel import CandidateResult

    scope = str(action.extra.get('lineage_id') or '')
    proposal = propose_policy_update(env.state, lineage_id=scope)
    wanted = str(action.extra.get("proposal_digest") or "")
    if proposal is None or (wanted and proposal["proposal_digest"] != wanted):
        return CandidateResult(status="no_policy_update",
                               extras={"reason": "the archive no longer supports this proposal"})
    accepted, reasons = evaluate_policy_update(env.state, proposal)
    candidate = dict(proposal["candidate"])
    row = {
        "kind": POLICY_KIND,
        "lineage_id": proposal['lineage_id'],
        "project_epoch": int(_get(env.state, 'project_epoch', 0) or 0),
        "id": f"policy-{proposal['proposal_digest'][:12]}",
        "old_policy_version": proposal["old_policy_version"],
        "observed_failure_patterns": proposal["observed_failure_patterns"],
        "proposed_policy_change": proposal["proposed_policy_change"],
        "reason": proposal["reason"],
        "accepted": accepted,
        "acceptance_reasons": reasons or ["scope, shape and archive replay passed"],
        "new_policy_version": candidate.get("version") if accepted else "",
        "installed": accepted,
        "policy": candidate if accepted else None,
        "proposal_digest": proposal["proposal_digest"],
        "scope": "current_workspace_and_lineage_only",
        "evaluation_basis": "generated_feedback_not_scientific_gain",
        "shared_installation": False,
        "epistemic": "GENERATED",
    }
    return CandidateResult(status="policy_installed" if accepted else "policy_rejected",
                           events=[make_event("ArtifactAdded", row, action_type=EVOLVE_POLICY,
                                              frontier_target_id=action.frontier_target_id)],
                           extras={"policy_update": row})


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_FAILURE_PHRASE = {
    "incomplete_idea": "it was not a complete idea",
    "unfalsifiable": "no outcome could refute it",
    "mechanism_unsupported": "its mechanism was not supported",
    "measurement_invalid": "its measurement did not answer the claim",
    "competing_explanation_unresolved": ("its mechanism was not distinguishable from a "
                                         "competing explanation"),
    "unrealistic_assumption": "it rested on an unrealistic assumption",
    "low_novelty": "verified literature or the critic found it already known",
    "scope_mismatch": "its scope made it untestable or trivial",
    "ill_posed_problem": "its problem was ill-posed",
    "missing_structure": "it borrowed a concept without a structural correspondence",
    "unclassified": "the critic found a blocking problem",
}


def discovery_summary(state: Any) -> dict[str, Any]:
    view = discovery_view(state)
    ideas, feedback = view["ideas"], view["feedback"]
    promising = [str(row["id"]) for row in ideas
                 if (feedback.get(str(row["id"])) or {}).get("verdict") == PROMISING]
    if promising:
        status = DISCOVERY_SUCCEEDED
    elif feedback:
        status = DISCOVERY_PARTIAL
    elif ideas:
        status = DISCOVERY_INCOMPLETE
    else:
        status = DISCOVERY_NOT_STARTED
    selected = view["selected_idea"]
    selected_status = dict(view["statuses"].get(selected) or {})
    by_id = {str(row["id"]): row for row in ideas}
    narrative: list[str] = []
    ordinal = {str(row["id"]): index + 1 for index, row in enumerate(ideas)}
    for row in ideas:
        idea_id = str(row["id"])
        fb = feedback.get(idea_id) or {}
        parent = by_id.get(str(row.get("parent_id") or ""))
        if parent is not None:
            jump = ("with a remote escape into a new conceptual frame"
                    if (row.get("remote_search") or {}).get("used")
                    else "without requiring another latent-space jump")
            narrative.append(
                f"The next hop used {row.get('operator')} and produced Idea {ordinal[idea_id]} "
                f"{jump}.")
        if fb.get("verdict") in {REVISE, COLLAPSE}:
            cause = normalize_failure(fb.get("root_failure_cause"))
            narrative.append(f"Idea {ordinal[idea_id]} {'collapsed' if fb['verdict'] == COLLAPSE else 'failed'} "
                             f"because {_FAILURE_PHRASE.get(cause, cause)}.")
            narrative.append("The failure was archived.")
        elif fb.get("verdict") == PROMISING:
            narrative.append(f"Idea {ordinal[idea_id]} was judged promising.")
    world_rows = [row for row in _get(state, "evidence_records", []) or []
                  if str(row.get("epistemic") or "") == "WORLD"]
    idea_ids = set(by_id)
    leaked = [row for row in world_rows
              if not row.get("attested") or row.get('promotion_refused')
              or str(row.get("evidence_id") or "") in idea_ids]
    evidence = str(selected_status.get("evidence_status") or "")
    headline = {
        DISCOVERY_SUCCEEDED: "Promising candidate formed; scientific validity is assessed separately.",
        DISCOVERY_PARTIAL: "Discovery produced validated ideas; none is judged promising yet.",
        DISCOVERY_INCOMPLETE: "Discovery produced ideas that have not been validated yet.",
        DISCOVERY_NOT_STARTED: "Discovery has not produced a complete idea.",
    }[status]
    if status == DISCOVERY_INCOMPLETE and evidence == "EMPIRICALLY_SUPPORTED":
        # "Not validated yet" is about the critic. Beside a READY verdict it read
        # as though nothing had been tested, when the idea had gone straight to
        # measurement and the measurement had come back.
        headline = ("The selected idea was tested directly and its registered prediction "
                    "held; no critic assessment of it was recorded.")
    tail = []
    if feedback:
        tail.append("The resulting research plan and experiment design were preserved.")
    if evidence != "EMPIRICALLY_SUPPORTED":
        tail.append("Empirical validation remains incomplete.")
    tail.append("No GENERATED claim was promoted to WORLD." if not leaked
                else f"{len(leaked)} WORLD rows lack kernel admission.")
    return {
        "rule": "discovery-summary-v1",
        "discovery_status": status,
        "headline": headline,
        "narrative": narrative,
        "closing": tail,
        "ideas": len(ideas),
        "validated": len(feedback),
        "promising": promising,
        "lineages": view["lineages"],
        "selected_idea": selected,
        "research_promise": selected_status.get("research_promise", "UNASSESSED"),
        "evidence_status": evidence or "UNVALIDATED",
        "evidence_route": selected_status.get("evidence_route", ""),
        "statuses": view["statuses"],
        "remote_searches": view["remote_searches"],
        "policy_version": view["policy_version"],
        "policy_history": view["policy_history"],
        "generated_promoted_to_world": len(leaked),
    }


# ---------------------------------------------------------------------------
# Research program reasoning. Data on ScientificState, not a second controller.
# Pairwise taste, discrimination, diagnosis and review modes change the next
# move. They are heuristic. They are not calibrated information gain.
# ---------------------------------------------------------------------------

LOCAL_RESEARCH = "LOCAL_RESEARCH"
PROGRAM_REVIEW = "PROGRAM_REVIEW"
NON_DISCRIMINATING = "NON_DISCRIMINATING"
FRONTIER_CAP = 5
LEDGER_CAP = 24
FAILURE_EXPLANATIONS = (
    "scientific_hypothesis_wrong",
    "assumption_wrong",
    "intervention_too_weak",
    "metric_insensitive",
    "confounder",
    "implementation_bug",
    "insufficient_sample",
    "world_instrument_invalid",
    "resource_truncation",
)


def branch_identity(question: str, mechanism: str, claim: str = "") -> str:
    """Identity is scientific content, not a code version."""
    return "br-" + digest({
        "question": " ".join(str(question or "").lower().split()),
        "mechanism": " ".join(str(mechanism or "").lower().split()),
        "claim": " ".join(str(claim or "").lower().split()),
    })[:16]


def experiment_discriminates(predictions: Mapping[str, Any] | None) -> bool:
    """Declared directional discrimination, not statistical power or a text comparison."""
    rows = list(dict(predictions or {}).values())
    if len(rows) < 2 or any(not isinstance(r, Mapping) for r in rows):
        return False
    if any(not isinstance(r.get('observable'), str) or not r['observable'].strip()
           or r.get('direction') not in {'increase', 'decrease', 'unchanged'} for r in rows):
        return False
    return (len({r['observable'].strip() for r in rows}) == 1
            and len({r['direction'] for r in rows}) > 1)


def compare_branches(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Pairwise comparison. Not an absolute self-score."""
    def pts(row: Mapping[str, Any]) -> dict[str, int]:
        return {
            "decisiveness": 1 if experiment_discriminates(row.get("predictions")) else 0,
            "observations_available": 1 if row.get('evidence_for') or row.get('evidence_against') else 0,
            "diagnosis_due": 1 if row.get('evidence_against') and not row.get('diagnosed') else 0,
            "feasibility": 0 if row.get("status") in {"killed", "refuted"} else 1,
        }
    a, b = pts(left), pts(right)
    reasons = [name for name in a if a[name] != b[name]]
    left_wins = sum(a.values()) > sum(b.values()) or (
        sum(a.values()) == sum(b.values()) and str(left.get("id") or "") < str(right.get("id") or ""))
    winner, loser = (left, right) if left_wins else (right, left)
    return {
        "winner": str(winner.get("id") or ""),
        "loser": str(loser.get("id") or ""),
        "reasons": reasons or ["tie_broken_by_identity"],
        "basis": "decision_readiness_heuristic; scientific_importance_and_novelty_unassessed",
    }


def pairwise_rank(branches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from itertools import combinations
    rows = [dict(item) for item in branches]
    score = {str(item.get("id") or ""): 0 for item in rows}
    comparisons = []
    for left, right in combinations(rows, 2):
        result = compare_branches(left, right)
        score[result["winner"]] = score.get(result["winner"], 0) + 1
        comparisons.append(result)
    ranked = sorted(rows, key=lambda item: (-score.get(str(item.get("id") or ""), 0),
                                            str(item.get("id") or "")))
    for item in ranked:
        item["pairwise_wins"] = score.get(str(item.get("id") or ""), 0)
    return ranked


def branch_priority(branch: Mapping[str, Any]) -> float:
    """Allocate diagnostic work; do not invent scientific value from field presence."""
    decisiveness = 1.0 if experiment_discriminates(branch.get("predictions")) else 0.25
    observations = 0.5 if branch.get('evidence_for') or branch.get('evidence_against') else 0.0
    diagnostic = 1.0 if branch.get('evidence_against') and not branch.get('diagnosed') else 0.0
    feasibility = 0.15 if branch.get("status") in {"killed", "refuted"} else 1.0
    spent = float(branch.get("resource_spent") or 0)
    budget = max(1.0, float(branch.get("budget") or 8))
    return ((decisiveness + observations + diagnostic) * feasibility) / (1.0 + spent / budget)


def _live_evidence(state: Any, idea_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in getattr(state, "evidence_records", []) or []
            if str(row.get("discovery_idea_id") or "") == idea_id
            and not row.get("invalidated")]


def internal_operational_validation(state: Any, row: Mapping[str, Any], feedback: Mapping[str, Any]) -> bool:
    """An internal operational question need not first be publication-worthy.

    This changes execution priority only; evidence admission and root-scope
    checks remain authoritative. Integrity or discrimination failures still
    require repair before a scientific pilot.
    """
    from .contract import contract_from_state
    from .experiment import canonical_claim, protocol_errors, OPERATIONS
    from .requirement import EXPERIMENT, acquired_world_id, compile_experiment_requirement
    idea = idea_of(row)
    protocol = idea.experiment or {}
    # Operational testability does not depend on novelty, critic approval or
    # whether a prediction is expected to be positive. Mission coverage is
    # checked separately by quality; this only permits a scoped experiment.
    if contract_from_state(state).target_quality != 'internal':
        return False
    req = compile_experiment_requirement(idea, idea_id=str(row.get('id') or ''),
                                         root_question=root_question(state))
    if req.kind != EXPERIMENT:
        return False
    if protocol.get('operation') in OPERATIONS:
        return bool(not protocol_errors(req)
                    and idea.central_claim.strip() == canonical_claim(protocol))
    # Admission establishes the instrument, not the broader scientific claim.
    # The existing bound-world matcher rechecks bytes and the complete trial
    # registration before permitting this scoped operational comparison.
    return bool(acquired_world_id(state, req))


def has_failure_diagnosis(state: Any, idea_id: str) -> bool:
    # Generated lists of possible explanations do not establish that the
    # observation was checked. Only the verified measurement can close this
    # prerequisite; the actual scientific cause may still be unresolved.
    return any(row.get("diagnosed") and not row.get("invalidated")
               and row.get("epistemic") == 'WORLD' and row.get('attested')
               and row.get("replay_ok") is True
               and str(row.get("discovery_idea_id") or "") == idea_id
               for row in getattr(state, "evidence_records", []) or [])


def failure_explanations_for(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    if not evidence:
        return ["world_instrument_invalid", "insufficient_sample"]
    if any(row.get("invalidated") or row.get("execution_status") != "ran"
           or row.get("replay_ok") is False for row in evidence):
        return ["implementation_bug", "world_instrument_invalid", "resource_truncation"]
    explanations = ["scientific_hypothesis_wrong", "assumption_wrong", "confounder"]
    if any(row.get("pilot_verdict") == "PILOT_INCONCLUSIVE" for row in evidence):
        explanations = ["intervention_too_weak", "metric_insensitive"] + explanations
    if any(int(row.get("sample_size") or 0) <= 1 for row in evidence):
        explanations.append("insufficient_sample")
    return explanations


def diagnose_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Separate reproducibility from scientific explanation; neither implies the other."""
    replayed = row.get("replay_ok") is True and not row.get("invalidated")
    negative = row.get("outcome") in {"weakens", "negative", "contradicted"}
    return {
        "evidence_id": row.get("evidence_id"),
        "registration_digest": row.get("registration_digest") or row.get("protocol_digest"),
        "world_digest": row.get("world_digest"),
        "status": "assessed_causes_unresolved" if replayed else "measurement_unverified",
        "observed": {k: row.get(k) for k in ("treatment", "control", "outcome", "sample_size")},
        "established": (["registered_observation_reproduced"] if replayed else []),
        "unresolved_explanations": failure_explanations_for([row]),
        "next_decision": ("compare_alternative_explanations_and_scope" if replayed and negative
                          else "design_a_discriminating_measurement" if replayed
                          else "repair_or_validate_measurement"),
        "scope": "registered_conditions_only; reproduction_does_not_identify_failure_cause",
        "epistemic": "GENERATED",
    }


def sync_research_frontier(state: Any, *, persist: bool = True) -> list[dict[str, Any]]:
    """Rebuild the small active frontier from scientific content + live evidence."""
    view = discovery_view(state)
    previous = {str(row.get("id") or ""): dict(row)
                for row in getattr(state, "research_branches", []) or []}
    built: list[dict[str, Any]] = []
    seen: set[str] = set()
    root = str(view.get("root") or getattr(state, "goal", "") or "")
    for row in view.get("ideas") or []:
        idea = idea_of(row)
        bid = branch_identity(idea.research_problem or root, idea.mechanism, idea.central_claim)
        if bid in seen:
            continue
        seen.add(bid)
        prev = previous.get(bid) or {}
        idea_id = str(row.get("id") or "")
        live = _live_evidence(state, idea_id)
        evidence_for = [str(item.get("evidence_id") or "") for item in live
                        if item.get("pilot_verdict") == "PILOT_POSITIVE"
                        or item.get("stage_verdict") == "POSITIVE"]
        evidence_against = [str(item.get("evidence_id") or "") for item in live
                            if item.get("pilot_verdict") == "PILOT_NEGATIVE"
                            or item.get("stage_verdict") == "NEGATIVE"]
        scoped = [b for tid, b in (getattr(state, 'belief_states', {}) or {}).items()
                  if any(t.get('id') == tid and t.get('discovery_idea_id') == idea_id
                         for t in getattr(state, 'theories', []))]
        killed_by = str(prev.get("killed_by") or "")
        if killed_by and any(item.get("evidence_id") == killed_by and item.get("invalidated")
                             for item in getattr(state, "evidence_records", []) or []):
            status = "active"
        elif str(prev.get("status") or "") == "killed":
            status = "killed"
        elif any(b.get('status') == 'refuted' for b in scoped):
            status = 'refuted'
        elif evidence_against and not evidence_for:
            status = "weakened"
        elif evidence_for:
            status = "active"
        else:
            status = str(prev.get("status") or "active")
        predictions = dict(idea.predictions)
        if predictions and not experiment_discriminates(predictions):
            next_experiment = NON_DISCRIMINATING
            uncertainty = "competing_explanations_make_the_same_prediction"
        else:
            next_experiment = str((idea.experiment or {}).get("operation") or idea.optional_experiment_direction or "")
            uncertainty = str(idea.uncertainties[0]) if idea.uncertainties else (
                "which_mechanism_explains_the_bound_world")
        commitment = next((a for a in getattr(state, "artifacts", []) or []
                           if a.get("kind") == "branch_commit" and a.get("branch_id") == bid), {})
        commit = dict(commitment.get("commit") or prev.get("commit") or {})
        built.append({
            "id": bid,
            "idea_id": idea_id,
            "research_question": idea.research_problem or root,
            "mechanism": idea.mechanism,
            "assumptions": list(idea.uncertainties or ()) or (
                [idea.competing_explanation] if idea.competing_explanation else []),
            "competing_explanations": (
                [idea.competing_explanation] if idea.competing_explanation else
                list(prev.get("competing_explanations") or [])),
            "predictions": predictions,
            "decisive_uncertainty": uncertainty,
            "next_discriminating_experiment": next_experiment,
            "evidence_for": [item for item in evidence_for if item],
            "evidence_against": [item for item in evidence_against if item],
            "resource_spent": sum(1 for item in getattr(state, "artifacts", []) or []
                                  if item.get("kind") == "action_attempt"
                                  and str(item.get("target") or "") in {idea_id, bid}),
            "budget": commit.get("resource_budget") or prev.get("budget") or 8,
            "kill_criteria": list(commit.get("kill_condition") or prev.get("kill_criteria")
                                  or ["two_negative_scientific_runs", "non_discriminating_after_diagnosis"]),
            "status": status,
            "commit": commit,
            "scientific_hypothesis": idea.central_claim,
            "diagnosed": has_failure_diagnosis(state, idea_id),
        })
    open_research = (getattr(state, 'contract', {}) or {}).get('research_mode', 'open_research') == 'open_research'
    # Keep every recorded branch visible to dependency selection. Pairwise
    # ordinal comparisons are not a calibrated basis for hiding a branch.
    frontier = built if open_research else pairwise_rank(built)[:FRONTIER_CAP]
    if persist:
        state.research_branches = frontier
        state.research_ledger = compact_ledger(state)
    return list(frontier)


def compact_ledger(state: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in getattr(state, "decision_journal", []) or []:
        rows.append({
            "question": item.get("question") or getattr(state, "goal", ""),
            "branch": item.get("branch") or "",
            "decision": item.get("decision") or item.get("kind") or "",
            "rationale": item.get("rationale") or item.get("reason") or "",
            "experiment": item.get("experiment") or item.get("change") or "",
            "evidence_id": item.get("evidence_id") or "",
            "result": item.get("result") or "",
            "interpretation": item.get("interpretation") or "",
            "continue": item.get("continue") or item.get("disposition") or "",
            "next_uncertainty": item.get("next_uncertainty") or "",
        })
    for item in getattr(state, "artifacts", []) or []:
        if item.get("kind") != "action_attempt":
            continue
        rows.append({
            "question": getattr(state, "goal", ""),
            "branch": item.get("target") or "",
            "decision": item.get("action_type") or "",
            "rationale": item.get("reason") or "",
            "experiment": item.get("variant") or "",
            "evidence_id": "",
            "result": item.get("outcome_key") or item.get("status") or "",
            "interpretation": "",
            "continue": "",
            "next_uncertainty": "",
        })
    return rows[-LEDGER_CAP:]


def program_review_trigger(state: Any) -> str:
    ticks = int(getattr(state, "ticks", 0) or 0)
    last = int((getattr(state, "metrics", {}) or {}).get("last_program_review_tick") or 0)
    if last and ticks <= last:
        return ""
    due = (not last) or (ticks - last >= 8)
    if getattr(state, "contradictions", None) and due:
        return "major_contradiction"
    if any(row.get("invalidated") for row in getattr(state, "evidence_records", []) or []) and due:
        return "evidence_invalidated"
    branches = list(getattr(state, "research_branches", []) or [])
    if any(row.get("status") in {"killed", "refuted"} for row in branches) and due:
        return "branch_killed"
    cheap = [row for row in getattr(state, "evidence_records", []) or []
             if row.get("pilot_verdict") == "PILOT_INCONCLUSIVE" and not row.get("invalidated")]
    if len(cheap) >= 2 and due:
        return "repeated_inconclusive"
    if ticks and ticks - last >= 8:
        return "checkpoint"
    return ""


def reason(state: Any) -> dict[str, Any]:
    """One researcher, two timescales. Not a MacroAgent."""
    branches = sync_research_frontier(state, persist=False)
    trigger = program_review_trigger(state)
    mode = PROGRAM_REVIEW if trigger else LOCAL_RESEARCH
    active = [row for row in branches if row.get("status") not in {"killed"}]
    open_research = (getattr(state, 'contract', {}) or {}).get('research_mode', 'open_research') == 'open_research'
    ranked = active if open_research else pairwise_rank(active)
    comparisons = []
    if len(ranked) >= 2:
        comparisons.append(compare_branches(ranked[0], ranked[1]))
    top = ranked[0] if ranked else {}
    if mode == PROGRAM_REVIEW:
        moves = [{
            "action": "REVIEW",
            "reason": "program_review",
            "target": str(top.get("idea_id") or getattr(state, "goal", "") or ""),
            "extra": {
                "program_review": True,
                "review_trigger": trigger,
                "branch_priority": 3.0,
                "review_answers": {
                    "core_question_still_central": None,
                    "importance_assessment": "not_established_by_execution_or_branch_rank",
                    "excluded": [row["id"] for row in branches if row.get("status") in {"killed", "refuted"}],
                    "deepen": [r['id'] for r in ranked if r.get('evidence_for') and not r.get('evidence_against')],
                    "reconsider": [r['id'] for r in ranked if r.get('evidence_against')],
                    "diagnose": [r['id'] for r in ranked if r.get('evidence_against') and not r.get('diagnosed')],
                    "kill": [],
                    "evidence_ids": sorted({eid for r in ranked for eid in
                                            r.get('evidence_for', []) + r.get('evidence_against', [])}),
                    "largest_uncertainty": str(top.get("decisive_uncertainty") or trigger),
                    "reformulate_root": None,
                    "decision": ("compare_explanations_before_continuing" if any(r.get('evidence_against') for r in ranked)
                                 else "test_unresolved_predictions"),
                    "scope": "review_is_a_decision_assessment_not_evidence_or_automatic_retirement",
                },
            },
        }]
        review = dict(moves[0]["extra"]["review_answers"])
    else:
        moves = []
        for top in (ranked if open_research else ranked[:1]):
            priority = 0.0 if open_research else branch_priority(top)
            if top.get("evidence_against") and not top.get("diagnosed"):
                moves.append({
                    "action": "VERIFY",
                    "reason": "diagnose_competing_failure_explanations",
                    "target": (top.get("evidence_against") or [""])[0],
                    "extra": {
                        "diagnostic": True,
                        "discovery_idea_id": top.get("idea_id"),
                        "branch_id": top.get("id"),
                        "failure_explanations": failure_explanations_for(
                            _live_evidence(state, str(top.get("idea_id") or ""))),
                        "branch_priority": max(2.2, priority + 1.0),
                    },
                })
            elif top.get("next_discriminating_experiment") == NON_DISCRIMINATING:
                moves.append({
                    "action": "IDEATE",
                    "reason": "need_discriminating_operationalization",
                    "target": str(top.get("idea_id") or ""),
                    "extra": {
                        "mode": EVOLVE, "operator": "DISCRIMINATIVE_TEST",
                        "must_change": "measurement",
                        "branch_id": top.get("id"),
                        "branch_priority": priority,
                    },
                })
        review = {}
    return {
        "mode": mode,
        "trigger": trigger,
        "branches": ranked,
        "comparison": comparisons,
        "moves": moves,
        "review": review,
        "next_uncertainty": str(top.get("decisive_uncertainty") or ""),
        "basis": "heuristic_research_reasoning_not_calibrated",
    }


def record_decision_outcome(state: Any, *, expected: str, actual: str, branch_id: str = "",
                            decision: str = "") -> dict[str, Any]:
    """Build an event payload; only the reducer may store the outcome."""
    return {
        "kind": "decision_outcome",
        "decision": decision,
        "branch": branch_id,
        "expected": expected,
        "actual": actual,
        "taste_error": bool(expected and actual and expected != actual),
        "tick": int(getattr(state, "ticks", 0) or 0) + 1,
    }


def freeze_branch_commit(branch: Mapping[str, Any], *, why: str = "") -> dict[str, Any]:
    existing = dict(branch.get("commit") or {})
    if existing:
        return existing
    return {
        "why_this_branch_matters": why or str(branch.get("scientific_hypothesis") or ""),
        "decisive_uncertainty": branch.get("decisive_uncertainty") or "",
        "expected_supporting_result": "observation matches primary prediction",
        "expected_refuting_result": "observation contradicts the registered prediction within its tested scope",
        "next_after_support": "cheap_then_frozen_confirmation_if_still_discriminating",
        "next_after_refute": "verify_observation_update_scoped_belief_then_reassess_question",
        "resource_budget": branch.get("budget") or 8,
        "kill_condition": list(branch.get("kill_criteria") or []),
        "review_condition": "after_key_evidence_or_eight_ticks",
        "amendments": [],
    }
