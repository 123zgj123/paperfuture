"""ResearchContract: the long-horizon identity of a research project.

A ``goal: str`` survives one action. It does not survive fifty. Over a long
trajectory the failure mode is not a wrong answer, it is a *substituted
question*: the root question becomes a sub-question, the sub-question becomes a
proxy, the proxy becomes whatever happens to be measurable, and nothing in the
state records that the original question was abandoned.

This module gives the project a contract that outlives any single hypothesis,
lineage, world, or design. The contract is written once at bootstrap and is
mutated only through explicit, recorded, contract-authorised transitions.

The root-alignment check here is a **lexical anchor guard**, not semantic
understanding. It cannot know that "throughput" and "requests per second" are
the same estimand. It is deliberately conservative: it flags a transition for
justification rather than silently accepting it, and a transition that cannot
name its link back to the root question is recorded as drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


CONTRACT_KEY = "research_contract"
RESEARCH_MODES = ("open_research", "metric_optimization")
# These fields express the user's mission, not worker-adjustable research choices.
IMMUTABLE_MISSION_FIELDS = frozenset({
    "root_question", "mission", "research_mode", "research_scope", "hard_constraints",
    "forbidden_scope_changes", "success_target", "success_metric", "required_outputs",
    "termination_requirements", "resource_limits", "world_selection", "allowed_reframes",
    "quality_requirements", "target_venue", "min_anchor_overlap", "max_instrumental_depth",
})

# Ordinal target quality. Not a venue prediction and not an accept probability.
QUALITY_TARGETS = ("exploratory", "internal", "workshop", "conference", "archival")

# Named research transitions a contract may authorise. Anything not listed is a
# scope change and needs an explicit contract amendment, not a quiet redirect.
# How the object of study may be chosen. Binding a world decides what the
# project's results will be *about*, so it is a scope decision the contract
# owns — not something a blocked run may settle for itself. A wrongly matched
# world still produces genuine WORLD evidence; it just answers a question
# nobody asked.
WORLD_SELECTION = ("declared_only", "catalog_match")

REFRAME_KINDS = (
    "mechanism_substitution",
    "measurement_repair",
    "representation_change",
    "population_restriction",
    "assumption_removal",
    "competing_explanation_test",
    "instrumental_subgoal",
)

_STOPWORDS = frozenset(
    """a an the of for to in on at by with without and or but is are was were be been
    being do does did doing can could should would will shall may might must this that
    these those it its as from into over under about than then when while which who whom
    what how why we our us you your they them their he she his her i me my not no if
    any some more most much many few less least very such own same so only just also
    does doesn don t s re ve ll d m""".split()
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_\-]*")


def anchor_terms(text: str) -> tuple[str, ...]:
    """Content terms of a research question, order preserved, deduplicated.

    Light suffix folding only (plurals / -ing / -ed). No stemmer, no embedding:
    a term match here is a lexical coincidence that licenses a check, never a
    claim that two questions ask the same thing.
    """
    out: list[str] = []
    for raw in _TOKEN.findall(str(text or "").casefold()):
        if raw in _STOPWORDS or len(raw) < 3:
            continue
        term = raw
        for suffix in ("ing", "ed", "es", "s"):
            if len(term) > len(suffix) + 3 and term.endswith(suffix):
                term = term[: -len(suffix)]
                break
        if term not in out:
            out.append(term)
    return tuple(out)


def _declared_mission_claim(value: Any) -> dict[str, Any]:
    """Keep a declared mission claim only if the kernel can render it.

    A declaration is a statement by the caller, not an inference about prose, so
    the check is whether it *is* a registered comparison — the kernel renders it
    and structures its own output. Anything the kernel cannot render is dropped
    rather than trusted; a dropped declaration leaves the mission exactly as
    unstructured as it was, which is the honest result.
    """
    if not isinstance(value, Mapping) or not value:
        return {}
    from .claim_relations import structured_from_execution

    claim = dict(value)
    return claim if structured_from_execution(claim) is not None else {}


@dataclass(frozen=True)
class ResearchContract:
    """What this project is for. Immutable except through recorded amendment."""

    root_question: str = ""
    research_mode: str = "open_research"
    research_scope: tuple[str, ...] = ()
    success_target: str = ""
    target_quality: str = "exploratory"
    target_venue: str = ""
    hard_constraints: tuple[str, ...] = ()
    allowed_reframes: tuple[str, ...] = REFRAME_KINDS
    forbidden_scope_changes: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()
    resource_limits: dict[str, float] = field(default_factory=dict)
    # What "better" means for this project, declared before any result exists.
    # Without it the project can judge single experiments and cannot say which
    # of its results is its best — see objective.py.
    success_metric: dict[str, Any] = field(default_factory=dict)
    quality_requirements: dict[str, str] = field(default_factory=dict)
    termination_requirements: tuple[str, ...] = ()
    # How far an instrumental sub-question may sit from the root before the
    # project is, in practice, researching something else.
    max_instrumental_depth: int = 2
    min_anchor_overlap: int = 2
    world_selection: str = "declared_only"
    # The mission stated as a registered comparison, when the caller can state
    # it. Prose has no structure this project will trust, and inferring one from
    # it is the boundary the design refuses to cross — so a live run measured
    # exactly what four missions asked, confirmed it, and could report nothing,
    # because the question was in English. Declaring is not inferring: this is
    # the caller's statement, rendered and structured by the kernel, kept only
    # if the kernel can render it.
    mission_claim: dict[str, Any] = field(default_factory=dict)
    amendments: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_question": self.root_question,
            "mission": self.mission,
            "research_mode": self.research_mode,
            "research_scope": list(self.research_scope),
            "success_target": self.success_target,
            "target_quality": self.target_quality,
            "target_venue": self.target_venue,
            "hard_constraints": list(self.hard_constraints),
            "allowed_reframes": list(self.allowed_reframes),
            "forbidden_scope_changes": list(self.forbidden_scope_changes),
            "required_outputs": list(self.required_outputs),
            "resource_limits": dict(self.resource_limits),
            "success_metric": dict(self.success_metric),
            "quality_requirements": dict(self.quality_requirements),
            "termination_requirements": list(self.termination_requirements),
            "max_instrumental_depth": int(self.max_instrumental_depth),
            "min_anchor_overlap": int(self.min_anchor_overlap),
            "world_selection": self.world_selection,
            "mission_claim": dict(self.mission_claim),
            "amendments": [dict(row) for row in self.amendments],
            "root_anchors": list(self.root_anchors),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResearchContract":
        row = dict(payload or {})

        def texts(key: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
            value = row.get(key)
            if value is None:
                return default
            if isinstance(value, str):
                value = [value]
            return tuple(str(item).strip() for item in value if str(item).strip())

        quality = str(row.get("target_quality") or "exploratory")
        return cls(
            root_question=str(row.get("root_question") or row.get("mission") or row.get("goal") or ""),
            research_mode=(str(row.get("research_mode")) if row.get("research_mode") in RESEARCH_MODES
                           else "open_research"),
            research_scope=texts("research_scope"),
            success_target=str(row.get("success_target") or ""),
            target_quality=quality if quality in QUALITY_TARGETS else "exploratory",
            target_venue=str(row.get("target_venue") or row.get("target_venue_if_any") or ""),
            hard_constraints=texts("hard_constraints"),
            allowed_reframes=texts("allowed_reframes", REFRAME_KINDS),
            forbidden_scope_changes=texts("forbidden_scope_changes"),
            required_outputs=texts("required_outputs"),
            resource_limits={
                str(k): float(v)
                for k, v in dict(row.get("resource_limits") or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
            quality_requirements={
                str(k): str(v) for k, v in dict(row.get("quality_requirements") or {}).items()
            },
            success_metric=dict(row.get("success_metric") or {}),
            termination_requirements=texts("termination_requirements"),
            max_instrumental_depth=max(0, int(row.get("max_instrumental_depth") or 2)),
            min_anchor_overlap=max(1, int(row.get("min_anchor_overlap") or 2)),
            world_selection=(str(row.get("world_selection") or "declared_only")
                             if str(row.get("world_selection") or "") in WORLD_SELECTION
                             else "declared_only"),
            mission_claim=_declared_mission_claim(row.get("mission_claim")),
            amendments=tuple(
                dict(item) for item in row.get("amendments") or [] if isinstance(item, Mapping)
            ),
        )

    @property
    def mission(self) -> str:
        """The immutable intent; ``root_question`` is its legacy serialized name."""
        return self.root_question

    @property
    def root_anchors(self) -> tuple[str, ...]:
        return anchor_terms(self.root_question)

    @property
    def is_set(self) -> bool:
        return bool(self.root_question)

    def required_overlap(self) -> int:
        anchors = self.root_anchors
        if not anchors:
            return 0
        return min(int(self.min_anchor_overlap), len(anchors))

    def amended(self, amendment: Mapping[str, Any]) -> "ResearchContract":
        """Record an authorised change. ``root_question`` is never replaced here.

        A project that wants a different root question is a different project;
        the honest move is a new contract, not a silent rewrite of this one.
        """
        row = dict(amendment)
        for key in IMMUTABLE_MISSION_FIELDS:
            row.pop(key, None)
        quality = row.get("target_quality")
        if quality is not None and (quality not in QUALITY_TARGETS
                or QUALITY_TARGETS.index(quality) < QUALITY_TARGETS.index(self.target_quality)):
            row.pop("target_quality", None)
        merged = self.to_dict()
        merged.pop("root_anchors", None)
        for key, value in row.items():
            if key in merged and key not in {"amendments"}:
                merged[key] = value
        merged["amendments"] = [dict(item) for item in self.amendments] + [dict(amendment)]
        return ResearchContract.from_dict(merged)


def contract_from_state(state: Any) -> ResearchContract:
    raw = state.get("contract") if isinstance(state, Mapping) else getattr(state, "contract", None)
    if isinstance(raw, Mapping) and (raw.get("root_question") or raw.get("mission")):
        return ResearchContract.from_dict(raw)
    goal = state.get("goal", "") if isinstance(state, Mapping) else getattr(state, "goal", "")
    return ResearchContract(root_question=str(goal or ""))


QUESTION_TRANSITION_FIELDS = (
    "parent_question_id", "triggering_evidence_ids", "triggering_blocker_ids",
    "invalidated_assumption", "rationale", "mission_link", "remaining_mission_gaps",
)


def validate_question_transition(state: Any, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Check recorded dependencies, never certify a proposal's scientific reasoning.

    A negative or limited observation can motivate a new question without
    satisfying any hypothesis. Only known, live observation/blocker references
    qualify; the mission-link prose remains an unverified GENERATED claim.
    """
    def rows(key: str) -> list[Mapping[str, Any]]:
        return list((state.get(key, []) if isinstance(state, Mapping) else getattr(state, key, [])) or [])

    qid = str(payload.get("id") or payload.get("question_id") or "")
    parent_id = str(payload.get("parent_question_id") or "")
    questions = {str(q.get("id") or ""): q for key in
                 ("open_questions", "resolved_questions", "partial_questions", "blocked_questions")
                 for q in rows(key)}
    if not qid or not parent_id or parent_id == qid or parent_id not in questions:
        return False, "question_transition_requires_known_distinct_parent"
    if qid in questions:
        return False, "question_identity_already_exists"
    if questions[parent_id].get("transition_status") == "reconsideration_required":
        return False, "parent_question_basis_requires_reconsideration"
    for key in ("text", "invalidated_assumption", "rationale", "mission_link"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            return False, f"question_transition_missing_{key}"
    gaps = payload.get("remaining_mission_gaps")
    if not isinstance(gaps, (list, tuple)) or not gaps or any(
            not isinstance(item, str) or not item.strip() for item in gaps):
        return False, "question_transition_requires_remaining_mission_gaps"
    normalize = lambda text: " ".join(re.findall(r"\w+", str(text).casefold()))
    proposed = normalize(payload["text"])
    if proposed in {normalize(questions[parent_id].get("text")), normalize(contract_from_state(state).mission)}:
        return False, "question_restatement_is_not_reformulation"
    evidence_ids = payload.get("triggering_evidence_ids", [])
    blocker_ids = payload.get("triggering_blocker_ids", [])
    for refs in (evidence_ids, blocker_ids):
        if not isinstance(refs, (list, tuple)) or any(
                not isinstance(ref, str) or not ref.strip() for ref in refs):
            return False, "question_transition_references_must_be_id_lists"
    if not evidence_ids and not blocker_ids:
        return False, "question_transition_requires_recorded_trigger"
    evidence = {str(e.get("evidence_id") or ""): e for e in rows("evidence_records")}
    from .kernel import refuse_generated_promotion
    for eid in evidence_ids:
        row = evidence.get(eid)
        if (not row or row.get("invalidated") or row.get("epistemic") != "WORLD"
                or not row.get("attested") or refuse_generated_promotion(row)):
            return False, f"question_transition_evidence_not_live:{eid}"
    blockers = {str(r.get("review_id") or ""): r for r in rows("reviews")
                if r.get("status") in {"open", "repairing"}
                and (r.get("blocking") or r.get("severity") in {"major", "fatal"})}
    blockers.update({str(q.get("id") or ""): q for q in rows("blocked_questions")
                     if q.get("blocked_reason") and q.get("blocked_reason") != "retired_by_project_pivot"})
    for bid in blocker_ids:
        if bid not in blockers:
            return False, f"question_transition_blocker_not_live:{bid}"
    instrumental = str(payload.get("instrumental_for") or "")
    if instrumental and (instrumental == qid or instrumental not in questions):
        return False, "question_transition_instrumental_parent_unknown"
    return True, "structurally_valid_mission_alignment_unknown"


@dataclass(frozen=True)
class RootAlignment:
    """Whether a research node still serves the contract's root question."""

    status: str  # "aligned" | "instrumental" | "drifted" | "unknown"
    shared_terms: tuple[str, ...] = ()
    missing_anchors: tuple[str, ...] = ()
    depth: int = 0
    reasons: tuple[str, ...] = ()

    @property
    def drifted(self) -> bool:
        return self.status == "drifted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "shared_terms": list(self.shared_terms),
            "missing_anchors": list(self.missing_anchors),
            "depth": self.depth,
            "reasons": list(self.reasons),
            "basis": "lexical_anchor_guard_not_semantic_equivalence",
        }


def root_alignment(
    contract: ResearchContract,
    *,
    text: str,
    justification: str = "",
    instrumental_for: str = "",
    depth: int = 0,
    reframe_kind: str = "",
) -> RootAlignment:
    """Does this question/claim still advance ``contract.root_question``?

    ``aligned``      shares enough root anchors on its own.
    ``instrumental`` does not, but names an aligned ancestor and a justification
                     that reaches back to the root vocabulary, within the
                     contract's instrumental depth.
    ``drifted``      neither. This is the "easier proxy became the project"
                     failure, and it is recorded rather than silently accepted.
    """
    anchors = contract.root_anchors
    if not anchors:
        return RootAlignment(status="unknown", reasons=("contract_has_no_root_question",))
    terms = set(anchor_terms(text))
    shared = tuple(term for term in anchors if term in terms)
    missing = tuple(term for term in anchors if term not in terms)
    reasons: list[str] = []

    blob = f"{text} {justification}".casefold()
    for forbidden in contract.forbidden_scope_changes:
        if forbidden.strip() and forbidden.strip().casefold() in blob:
            return RootAlignment(
                status="drifted",
                shared_terms=shared,
                missing_anchors=missing,
                depth=depth,
                reasons=(f"forbidden_scope_change:{forbidden}",),
            )
    if reframe_kind and contract.allowed_reframes and reframe_kind not in contract.allowed_reframes:
        reasons.append(f"reframe_kind_not_authorised:{reframe_kind}")

    if len(shared) >= contract.required_overlap() and not reasons:
        return RootAlignment(status="aligned", shared_terms=shared, missing_anchors=missing, depth=depth)

    if depth > int(contract.max_instrumental_depth):
        reasons.append(f"instrumental_depth_{depth}_exceeds_{contract.max_instrumental_depth}")
        return RootAlignment(
            status="drifted", shared_terms=shared, missing_anchors=missing,
            depth=depth, reasons=tuple(reasons),
        )
    if not instrumental_for:
        reasons.append("no_aligned_parent_declared")
    justification_terms = set(anchor_terms(justification))
    if not (justification_terms & set(anchors)):
        reasons.append("justification_does_not_reach_root_question")
    if reasons:
        return RootAlignment(
            status="drifted", shared_terms=shared, missing_anchors=missing,
            depth=depth, reasons=tuple(reasons),
        )
    return RootAlignment(
        status="instrumental", shared_terms=shared, missing_anchors=missing, depth=depth,
        reasons=("serves_root_question_through_declared_parent",),
    )


def alignment_of_question(
    contract: ResearchContract,
    question: Mapping[str, Any],
    *,
    known: Iterable[Mapping[str, Any]] = (),
) -> RootAlignment:
    """Alignment of one recorded question, resolving its instrumental chain."""
    rows = {str(row.get("id") or ""): dict(row) for row in known if isinstance(row, Mapping)}
    depth, cursor, seen = 0, dict(question), {str(question.get("id") or "")}
    parent = str(cursor.get("instrumental_for") or "")
    while parent and parent in rows and parent not in seen and depth < 16:
        seen.add(parent)
        depth += 1
        cursor = rows[parent]
        parent = str(cursor.get("instrumental_for") or "")
    return root_alignment(
        contract,
        text=str(question.get("text") or question.get("claim") or ""),
        justification=str(question.get("root_link") or question.get("why_this_advances_root") or ""),
        instrumental_for=str(question.get("instrumental_for") or ""),
        depth=depth,
        reframe_kind=str(question.get("reframe_kind") or ""),
    )
