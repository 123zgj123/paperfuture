"""ResearchQualityState and ReviewRecord: how good is this research project?

``ScientificState`` answers *what do we know*. It has no way to say *this
project is not good enough yet, and here is exactly which part is not good
enough*. Without that, the controller cannot tell a finished project from an
unfinished one, cannot tell a real objection from a wording complaint, and
cannot notice that ten successful actions moved no quality dimension at all.

Two rules keep this honest:

1. **Ordinal, never a fabricated probability.** A dimension is
   ``unknown / missing / weak / partial / adequate``. There is no
   ``accept_probability = 0.81`` here, because nothing in this runtime has ever
   been calibrated against real acceptance decisions.
2. **Derived, never asserted.** Quality is recomputed from admitted evidence,
   recorded reviews and the contract on every replay. A worker cannot write
   "experimental_adequacy: adequate" into the state; it can only produce
   evidence or an objection, and the projection follows.

Reviewer judgment is a *planning signal*, not evidence. A review can lower a
dimension and open repair debt. It can never raise ``central_claim_support``,
because agreement is not measurement (see ``social_consensus_is_not_evidence``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..validity import (
    ABLATION, BLOCKING, CLAIM, EXECUTION, FAIL, OBJECT, ORACLE, PASS, PLACEBO, REGISTRATION,
    SAMPLE, UNDEFINED, check_reason, check_status, corroborates, sample_floor,
)
from .claim_relations import (
    EQUIVALENT,
    INCOMPATIBLE,
    RULE_VERSION as RELATION_RULE_VERSION,
    UNKNOWN as RELATION_UNKNOWN,
    invert_outcome,
    relate,
    structured_claim,
    structured_from_execution,
)
from .contract import ResearchContract, alignment_of_question, contract_from_state
from .study import aggregate as study_aggregate, study_of


# --------------------------------------------------------------------------
# Ordinal quality scale
# --------------------------------------------------------------------------

STATUS_ORDER = ("unknown", "missing", "weak", "partial", "adequate")
STATUS_RANK = {name: index for index, name in enumerate(STATUS_ORDER)}

DIMENSIONS = (
    "novelty",
    "importance",
    "technical_correctness",
    "experimental_adequacy",
    "baseline_strength",
    "measurement_validity",
    "mechanism_support",
    "robustness",
    "generalization",
    "replication",
    "literature_coverage",
    "central_claim_support",
)

# Dimensions no execution can settle. Only literature and review evidence speak
# to them, so they stay ``unknown`` rather than being invented from a run.
JUDGMENT_DIMENSIONS = frozenset({"novelty", "importance"})

# What each validity check speaks to. The structural reviewer routes its
# objections through this same table, so a failed check lowers one dimension
# and raises one objection, and the two cannot disagree about which.
CHECK_DIMENSION = {
    REGISTRATION: "technical_correctness",
    EXECUTION: "technical_correctness",
    OBJECT: "baseline_strength",
    ABLATION: "baseline_strength",
    ORACLE: "measurement_validity",
    CLAIM: "measurement_validity",
    SAMPLE: "experimental_adequacy",
    PLACEBO: "robustness",
}

DEFAULT_REQUIREMENTS = {
    "exploratory": {"literature_coverage": "partial", "technical_correctness": "partial"},
    "internal": {
        "literature_coverage": "partial", "technical_correctness": "adequate",
        "measurement_validity": "partial", "experimental_adequacy": "partial",
        "central_claim_support": "partial",
    },
    "workshop": {
        "literature_coverage": "adequate", "technical_correctness": "adequate",
        "measurement_validity": "adequate", "experimental_adequacy": "partial",
        "baseline_strength": "partial", "central_claim_support": "partial",
        "novelty": "partial",
    },
    "conference": {
        "literature_coverage": "adequate", "technical_correctness": "adequate",
        "measurement_validity": "adequate", "experimental_adequacy": "adequate",
        "baseline_strength": "adequate", "mechanism_support": "partial",
        "robustness": "partial", "replication": "partial",
        "central_claim_support": "adequate", "novelty": "partial",
        "importance": "partial",
    },
    "archival": {
        "literature_coverage": "adequate", "technical_correctness": "adequate",
        "measurement_validity": "adequate", "experimental_adequacy": "adequate",
        "baseline_strength": "adequate", "mechanism_support": "adequate",
        "robustness": "adequate", "generalization": "partial",
        "replication": "adequate", "central_claim_support": "adequate",
        "novelty": "partial", "importance": "partial",
    },
}


def meets(status: str, requirement: str) -> bool:
    return STATUS_RANK.get(str(status), 0) >= STATUS_RANK.get(str(requirement), 0)


def cap(status: str, ceiling: str) -> str:
    return status if STATUS_RANK.get(status, 0) <= STATUS_RANK.get(ceiling, 0) else ceiling


def requirements_for(contract: ResearchContract) -> dict[str, str]:
    base = dict(DEFAULT_REQUIREMENTS.get(contract.target_quality, DEFAULT_REQUIREMENTS["exploratory"]))
    for name, value in (contract.quality_requirements or {}).items():
        if name in DIMENSIONS and str(value) in STATUS_RANK:
            base[name] = str(value)
    return base


# --------------------------------------------------------------------------
# Reviews
# --------------------------------------------------------------------------

SEVERITIES = ("fatal", "major", "minor")

REPAIR_TYPES = (
    "writing_only",
    "more_evidence",
    "stronger_baseline",
    "broader_evaluation",
    "measurement_redesign",
    "experiment_redesign",
    "causal_redesign",
    "assumption_reframe",
    "representation_change",
    "project_pivot",
)

# A repair that changes only how the work is described. Everything else needs a
# world to answer it.
PRESENTATION_REPAIRS = frozenset({"writing_only"})

# Repairs that cannot be discharged by running the same design again.
REDESIGN_REPAIRS = frozenset(
    {
        "measurement_redesign",
        "experiment_redesign",
        "causal_redesign",
        "assumption_reframe",
        "representation_change",
    }
)

PIVOT_REPAIRS = frozenset({"project_pivot"})

# Resolution kinds a reviewer-facing worker might propose. The first group is
# text; the second is work. Only the second can close an evidence-requiring
# objection.
CAVEAT_ONLY_RESOLUTIONS = frozenset(
    {
        "rewrite",
        "reword",
        "claim_narrowing",
        "added_caveat",
        "added_limitation",
        "added_discussion",
        "better_wording",
        "clarification",
        "writing_only",
    }
)

EVIDENCE_RESOLUTIONS = frozenset(
    {
        "new_world_evidence",
        "new_valid_experiment",
        "replication",
        "baseline_improvement",
        "measurement_repair",
        "design_repair",
    }
)

REPAIR_DIMENSION = {
    "stronger_baseline": "baseline_strength",
    "broader_evaluation": "generalization",
    "measurement_redesign": "measurement_validity",
    "experiment_redesign": "experimental_adequacy",
    "causal_redesign": "mechanism_support",
    "more_evidence": "central_claim_support",
    "assumption_reframe": "technical_correctness",
    "representation_change": "technical_correctness",
    "project_pivot": "importance",
    "writing_only": "technical_correctness",
}


@dataclass(frozen=True)
class ReviewRecord:
    """A structured objection that is allowed to change the research plan."""

    review_id: str
    severity: str = "minor"
    dimension: str = "technical_correctness"
    objection: str = ""
    affected_claim: str = ""
    affected_design: str = ""
    repair_type: str = "writing_only"
    requires_new_evidence: bool = False
    status: str = "open"  # open | repairing | resolved | withdrawn | superseded
    question_id: str = ""
    theory_id: str = ""
    raised_at_tick: int = 0
    resolution_kind: str = ""
    resolution_evidence_id: str = ""
    resolved_at_tick: int = 0
    source: str = "reviewer_judgment_not_evidence"
    # An assessment records a judgment about a dimension no execution can
    # settle. It is not an objection, and it never corroborates a claim.
    assessment: str = ""
    supporting_refs: tuple[str, ...] = ()
    assessment_basis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "severity": self.severity,
            "dimension": self.dimension,
            "objection": self.objection,
            "affected_claim": self.affected_claim,
            "affected_design": self.affected_design,
            "repair_type": self.repair_type,
            "requires_new_evidence": bool(self.requires_new_evidence),
            "status": self.status,
            "question_id": self.question_id,
            "theory_id": self.theory_id,
            "raised_at_tick": int(self.raised_at_tick),
            "resolution_kind": self.resolution_kind,
            "resolution_evidence_id": self.resolution_evidence_id,
            "resolved_at_tick": int(self.resolved_at_tick),
            "source": self.source,
            "assessment": self.assessment,
            "supporting_refs": list(self.supporting_refs),
            "assessment_basis": self.assessment_basis,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ReviewRecord":
        row = dict(payload or {})
        severity = str(row.get("severity") or "minor").lower()
        repair = str(row.get("repair_type") or "writing_only")
        dimension = str(row.get("dimension") or "technical_correctness")
        return cls(
            review_id=str(row.get("review_id") or row.get("id") or ""),
            severity=severity if severity in SEVERITIES else "minor",
            dimension=dimension if dimension in DIMENSIONS else "technical_correctness",
            objection=str(row.get("objection") or ""),
            affected_claim=str(row.get("affected_claim") or ""),
            affected_design=str(row.get("affected_design") or ""),
            repair_type=repair if repair in REPAIR_TYPES else "writing_only",
            requires_new_evidence=bool(row.get("requires_new_evidence")),
            status=str(row.get("status") or "open"),
            question_id=str(row.get("question_id") or ""),
            theory_id=str(row.get("theory_id") or ""),
            raised_at_tick=int(row.get("raised_at_tick") or 0),
            resolution_kind=str(row.get("resolution_kind") or ""),
            resolution_evidence_id=str(row.get("resolution_evidence_id") or ""),
            resolved_at_tick=int(row.get("resolved_at_tick") or 0),
            source=str(row.get("source") or "reviewer_judgment_not_evidence"),
            assessment=str(row.get("assessment") or "") if str(
                row.get("assessment") or "") in STATUS_RANK else "",
            supporting_refs=tuple(str(item) for item in row.get("supporting_refs") or []),
            assessment_basis=str(row.get("assessment_basis") or ""),
        )


def normalize_review(payload: Mapping[str, Any], *, tick: int = 0) -> dict[str, Any]:
    """Admit a proposed review into canonical form.

    ``requires_new_evidence`` is not a reviewer's opinion to hand out freely: any
    repair type other than pure presentation *is* an evidence-requiring
    objection, so the flag is forced true. A reviewer cannot mark a
    ``measurement_redesign`` as answerable by rewriting.
    """
    record = ReviewRecord.from_dict(payload)
    needs_world = record.repair_type not in PRESENTATION_REPAIRS
    if record.severity == "fatal":
        needs_world = True
    row = record.to_dict()
    if (record.source == 'structural_result_review' and payload.get('result_review_scope')
            and not record.objection and record.severity == 'minor'
            and record.repair_type == 'writing_only'):
        row.update(status='resolved', requires_new_evidence=False,
                   resolution_kind='result_scope_reviewed',
                   result_review_scope=str(payload['result_review_scope']))
        return row
    if (payload.get("program_review") and payload.get('dimension') == 'program_coherence'
            and record.severity == 'minor' and record.repair_type == 'writing_only'
            and record.objection in {'', 'program review of competing branches'}):
        # A checkpoint is a recorded decision, not an empty scientific objection.
        row.update(program_review=True, dimension="program_coherence",
                   answers=dict(payload.get("answers") or {}),
                   trigger=str(payload.get("trigger") or ""),
                   objection=str(payload.get("objection") or payload.get("text") or ""),
                   status="resolved", resolution_kind="program_review_recorded",
                   requires_new_evidence=False,
                   raised_at_tick=int(payload.get("raised_at_tick") or tick or 0))
        return row
    if record.assessment and not record.objection and record.severity != 'fatal':
        # A rating with no complaint attached is a judgment, not review debt.
        row["status"] = "resolved"
        row["requires_new_evidence"] = False
        row["resolution_kind"] = "assessment_recorded"
        row["review_id"] = row["review_id"] or f"assessment-{record.dimension}"
        if payload.get('source') == 'structural_result_review' and payload.get('result_review_scope'):
            row['result_review_scope'] = str(payload['result_review_scope'])
        if isinstance(payload.get('panel'), Mapping):
            # What a panel judged, who sat on it and where they disagreed. Kept
            # with the grade so a grade can never be read without them.
            row['panel'] = dict(payload['panel'])
        return row
    row.update(status='open', resolution_kind='', resolution_evidence_id='', resolved_at_tick=0)
    row["requires_new_evidence"] = bool(record.requires_new_evidence or needs_world)
    row["raised_at_tick"] = int(payload.get("raised_at_tick") or tick or 0)
    if not row["review_id"]:
        row["review_id"] = f"review-{row['dimension']}-{row['raised_at_tick']}"
    return row


def resolution_admissible(
    review: Mapping[str, Any],
    *,
    resolution_kind: str,
    evidence_id: str = "",
    evidence_rows: Iterable[Mapping[str, Any]] = (),
    theories: Iterable[Mapping[str, Any]] = (),
    design_revisions: Iterable[Mapping[str, Any]] = (),
    min_sample: float = 0.0,
) -> tuple[bool, str]:
    """Deterministic caveat-only guard.

    If an objection requires new evidence, prose cannot close it. Not a rewrite,
    not a narrower claim, not another caveat, not a new discussion section. The
    only admissible resolutions are new admitted WORLD evidence, a replication,
    a repaired measurement, a repaired design, or a stronger baseline — each
    identified by an evidence row that actually exists and actually promoted.
    """
    record = ReviewRecord.from_dict(review)
    kind = str(resolution_kind or "").strip().lower()
    if not record.requires_new_evidence:
        return True, ""
    if kind in CAVEAT_ONLY_RESOLUTIONS or not kind:
        return False, (
            f"caveat_only_repair_refused: {record.review_id} requires new evidence; "
            f"'{kind or 'unspecified'}' only changes the text"
        )
    if kind not in EVIDENCE_RESOLUTIONS:
        return False, f"unknown_resolution_kind:{kind}"
    eid = str(evidence_id or "")
    if not eid:
        return False, f"resolution_requires_admitted_evidence_id: {record.review_id}"
    row = next(
        (
            dict(item)
            for item in evidence_rows
            if str(item.get("evidence_id") or "") == eid
        ),
        None,
    )
    if row is None:
        return False, f"resolution_evidence_not_in_state: {eid}"
    if not (row.get("promotion_ok") and str(row.get("epistemic") or "") == "WORLD" and row.get("attested")):
        return False, f"resolution_evidence_did_not_promote_to_world: {eid}"
    if row.get('invalidated') or row.get('promotion_refused'):
        return False, f"resolution_evidence_invalidated: {eid}"
    if record.theory_id and row.get('theory_id') != record.theory_id:
        revised = next((t for t in theories if t.get('id') == row.get('theory_id')), {})
        revision = next((r for r in design_revisions if r.get('revision_id') == revised.get('revision_id')
                         and r.get('parent_design') == record.theory_id
                         and record.review_id in (r.get('addresses_reviews') or [])), None)
        if not revision:
            return False, f"resolution_evidence_unrelated_to_theory: {eid}"
    if record.question_id and row.get('question_id') != record.question_id:
        return False, f"resolution_evidence_unrelated_to_question: {eid}"
    admitted = row.get("admitted_at_tick")
    if isinstance(admitted, int) and admitted < record.raised_at_tick:
        return False, f"resolution_evidence_predates_objection: {eid}"
    from ..validity import assess_probe, MEASUREMENT_CHECKS
    assessment = assess_probe(row, min_sample=min_sample)
    required_checks = set(MEASUREMENT_CHECKS)
    required_checks.update(k for k, dimension in CHECK_DIMENSION.items()
                           if dimension == record.dimension)
    if any(assessment.check(name).blocks for name in required_checks):
        return False, f"resolution_evidence_does_not_repair_{record.dimension}: {eid}"
    if row.get('study_stage') in {'pilot', 'calibration', 'scientific_pilot'}:
        return False, f"resolution_requires_final_evidence: {eid}"
    if record.dimension in JUDGMENT_DIMENSIONS:
        return False, f"resolution_requires_substantive_reassessment: {record.dimension}"
    if kind == 'replication' or record.repair_type == 'broader_evaluation':
        prior = next((r for r in evidence_rows if r.get('evidence_id') == row.get('replication_of')), None)
        if (not row.get('independent_replication') or not prior
                or prior.get('world_digest') == row.get('world_digest')
                or prior.get('theory_id') != row.get('theory_id')):
            return False, f"resolution_requires_independent_replication: {eid}"
    return True, ""


def open_reviews(rows: Iterable[Mapping[str, Any]], *, severity: str = "") -> list[dict[str, Any]]:
    out = []
    for row in rows or []:
        record = ReviewRecord.from_dict(row)
        if record.status in {"resolved", "withdrawn", "superseded"}:
            continue
        if severity and record.severity != severity:
            continue
        out.append({**dict(row), **record.to_dict()})
    return out


def review_debt(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    rows = list(rows or [])
    return {
        "fatal_objections": len(open_reviews(rows, severity="fatal")),
        "major_objections": len(open_reviews(rows, severity="major")),
        "minor_objections": len(open_reviews(rows, severity="minor")),
        "evidence_requiring_objections": sum(
            1 for row in open_reviews(rows) if row.get("requires_new_evidence")
        ),
        "redesign_objections": sum(
            1 for row in open_reviews(rows) if row.get("repair_type") in REDESIGN_REPAIRS
        ),
        "pivot_objections": sum(
            1 for row in open_reviews(rows) if row.get("repair_type") in PIVOT_REPAIRS
        ),
    }


# --------------------------------------------------------------------------
# Quality projection
# --------------------------------------------------------------------------


@dataclass
class QualityDimension:
    name: str
    status: str = "unknown"
    blocking_reasons: list[str] = field(default_factory=list)
    supporting_refs: list[str] = field(default_factory=list)
    required_next_work: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "blocking_reasons": list(dict.fromkeys(self.blocking_reasons)),
            "supporting_refs": list(dict.fromkeys(self.supporting_refs)),
            "required_next_work": list(dict.fromkeys(self.required_next_work)),
        }


def _world_rows(state: Any) -> list[dict[str, Any]]:
    from ..validity import evidence_eligible
    retired = {str(t.get("id") or "") for t in getattr(state, "theories", []) or []
               if t.get("status") == "retired"}
    return [
        dict(row)
        for row in getattr(state, "evidence_records", []) or []
        if evidence_eligible(row) and str(row.get("theory_id") or "") not in retired
        and row.get("attested")
        and row.get("promotion_ok")
    ]


def _probe_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("role") or "") == "ProbeEvidence"]


def _replication_possible(state: Any) -> bool:
    """Is there a heldout world a registered script could actually read?

    A run bound to a primary whose heldout shares no payload filename is asked
    for a replication no script can perform. Reporting that as "not attempted"
    invites another attempt; reporting it as impossible names what would fix it.
    """
    from pathlib import Path as _Path

    def payloads(row: Mapping[str, Any]) -> set[str]:
        skip = {"seed.json", "origin.json", "manifest.json"}
        return {_Path(str(name)).name for name in row.get("files") or []
                if _Path(str(name)).name not in skip}

    rows = list(getattr(state, "world_versions", []) or [])
    primary = next((row for row in rows
                    if str(row.get("id") or row.get("world_id") or "")
                    == str(getattr(state, "world_id", ""))), {})
    if not primary or not payloads(primary):
        return True
    for row in rows:
        if row.get("research_role") != "heldout":
            continue
        if not payloads(row):
            return True
        if primary.get("schema") == row.get("schema") and payloads(primary) & payloads(row):
            return True
    return False


def _min_sample(contract: ResearchContract) -> float:
    return sample_floor(contract.resource_limits)


def result_review_scope(state: Any) -> dict[str, Any]:
    """Bind review coverage to the actual conclusions, measurements and contract."""
    from .experiment import digest
    retired = {t.get('id') for t in state.theories if t.get('status') == 'retired'}
    rows = [r for r in state.evidence_records if r.get('role') == 'ProbeEvidence'
            and r.get('epistemic') == 'WORLD' and r.get('attested') and not r.get('invalidated')
            and r.get('theory_id') not in retired
            and r.get('study_stage') not in {'pilot', 'calibration', 'scientific_pilot'}]
    facts = [{k: r.get(k) for k in ('evidence_id', 'world_digest', 'experiment_digest',
             'registration_digest', 'outcome', 'validity', 'promotion_ok', 'reproduction_ok',
             'identity_ok', 'transfer_ok', 'inference_link')} for r in rows]
    contract = contract_from_state(state)
    return {'evidence_ids': sorted(str(r['evidence_id']) for r in rows),
            'digest': digest({'facts': sorted(facts, key=lambda r: str(r['evidence_id'])),
                              'root': contract.root_question, 'quality': contract.target_quality,
                              'requirements': contract.quality_requirements,
                              'epoch': state.project_epoch})}


def compute_quality(state: Any) -> dict[str, Any]:
    """Recompute the project's quality state from canonical facts only.

    Every branch here reads admitted evidence, recorded reviews, surveys, or the
    contract. Nothing reads a model's self-assessment.
    """
    contract = contract_from_state(state)
    reviews = list(getattr(state, "reviews", []) or [])
    world = _world_rows(state)
    probes = _probe_rows(world)
    surveys = [
        dict(row)
        for row in getattr(state, "artifacts", []) or []
        if row.get("kind") == "literature_survey" and row.get("works")
    ]
    theories = [dict(row) for row in getattr(state, "theories", []) or []
                if row.get("status") != "retired"]
    verified = dict(getattr(state, "verified_research_state", {}) or {})
    contradictions = list(getattr(state, "contradictions", []) or [])
    min_sample = _min_sample(contract)

    dims = {name: QualityDimension(name=name) for name in DIMENSIONS}

    def note(name: str, status: str, *, reason: str = "", ref: str = "", work: str = "") -> None:
        row = dims[name]
        if STATUS_RANK.get(status, 0) > STATUS_RANK.get(row.status, 0) or row.status == "unknown":
            row.status = status
        if reason:
            row.blocking_reasons.append(reason)
        if ref:
            row.supporting_refs.append(ref)
        if work:
            row.required_next_work.append(work)

    def force(name: str, status: str, *, reason: str = "", work: str = "") -> None:
        row = dims[name]
        row.status = status
        if reason:
            row.blocking_reasons.append(reason)
        if work:
            row.required_next_work.append(work)

    # -- literature coverage -------------------------------------------------
    if not surveys:
        force("literature_coverage", "missing", reason="no_verified_literature_survey",
              work="SURVEY")
    else:
        hypothesis_checked = {str(row.get("theory_id") or "") for row in surveys}
        substantive = [t for t in theories if t.get("role") != "competing_explanation"]
        unchecked = [str(t.get("id") or "") for t in substantive if str(t.get("id") or "") not in hypothesis_checked]
        if not substantive:
            note("literature_coverage", "partial", reason="survey_without_hypothesis_specific_check",
                 work="THEORIZE")
        elif unchecked:
            note("literature_coverage", "partial",
                 reason=f"hypotheses_without_literature_check:{len(unchecked)}", work="SURVEY")
        else:
            note("literature_coverage", "adequate",
                 ref=str(surveys[-1].get("id") or "literature_survey"))

    # -- technical correctness ----------------------------------------------
    # Only executed experiments speak to this. A descriptive observation carries
    # ``cannot_corroborate`` by construction — that is an honest label meaning
    # "this describes the dataset, it does not test the hypothesis", not a
    # validation failure — and reading it as a defect demanded a redesign of a
    # design that did not exist yet.
    # Each dimension below reads the checks `CHECK_DIMENSION` assigns it and
    # nothing else, so a defect is charged once, under its own name — and a
    # dimension is adequate only when its checks passed, never because a
    # failure flag happened to be absent.
    broken = sorted({f"{name}: {check_reason(row, name)}"
                     for row in probes for name in (REGISTRATION, EXECUTION)
                     if check_status(row, name) in BLOCKING})
    if not probes:
        force("technical_correctness", "unknown", reason="no_executed_experiment",
              work="PROBE")
    elif broken:
        force("technical_correctness", "weak", reason="; ".join(broken), work="REDESIGN")
    else:
        note("technical_correctness", "adequate", ref=str(world[-1].get("evidence_id") or ""))
    if contradictions:
        force("technical_correctness", cap(dims["technical_correctness"].status, "weak"),
              reason=f"unresolved_contradictions:{len(contradictions)}", work="VERIFY")

    # -- experimental adequacy (power / sample) ------------------------------
    if not probes:
        force("experimental_adequacy", "missing", reason="no_executed_experiment", work="PROBE")
    else:
        underpowered = [row for row in probes if check_status(row, SAMPLE) == FAIL]
        unsized = [row for row in probes if check_status(row, SAMPLE) == UNDEFINED]
        if underpowered:
            force("experimental_adequacy", "weak",
                  reason=(
                      f"underpowered:{len(underpowered)}_of_{len(probes)}_below_"
                      f"contract_min_sample_{min_sample:g}"
                  ),
                  work="REDESIGN")
        elif unsized:
            # A result whose power nobody reported is not an adequate result.
            # The repair is for the experiment to report how many units it
            # measured, which is a measurement change, not another design.
            refused = [row for row in unsized
                       if str(row.get("sample_size_basis") or "").startswith("refused")]
            force("experimental_adequacy", "weak",
                  reason=("reported_sample_exceeded_the_attested_world" if refused
                          else f"experiment_does_not_report_units_measured:{len(unsized)}"),
                  work="REDESIGN")
        else:
            note("experimental_adequacy", "adequate", ref=str(probes[-1].get("evidence_id") or ""))

    # -- baseline strength ---------------------------------------------------
    if not probes:
        force("baseline_strength", "missing", reason="no_control_arm_executed", work="PROBE")
    else:
        needs_ablation = [row for row in probes if check_status(row, ABLATION) == FAIL]
        degenerate = [row for row in probes if check_status(row, OBJECT) == FAIL]
        if degenerate:
            force("baseline_strength", "missing",
                  reason="control_and_treatment_measured_nothing", work="REDESIGN")
        elif needs_ablation:
            force("baseline_strength", "weak",
                  reason="competing_mechanism_ablation_not_registered", work="REDESIGN")
        else:
            note("baseline_strength", "partial", ref=str(probes[-1].get("evidence_id") or ""))
            if any(row.get("mechanism_validated") for row in probes):
                note("baseline_strength", "adequate")

    # -- measurement validity ------------------------------------------------
    if not probes:
        force("measurement_validity", "missing", reason="nothing_measured", work="PROBE")
    else:
        blind = [row for row in probes if check_status(row, ORACLE) == FAIL]
        unmeasured = [row for row in probes if check_status(row, ORACLE) == UNDEFINED]
        unclaimed = [row for row in probes if check_status(row, CLAIM) in BLOCKING]
        if blind:
            force("measurement_validity", "missing",
                  reason="measure_does_not_respond_to_the_oracle", work="REDESIGN")
        elif unmeasured:
            force("measurement_validity", "weak",
                  reason="oracle_placebo_required_but_not_measured", work="REDESIGN")
        elif unclaimed:
            force("measurement_validity", "weak",
                  reason="result_is_not_bound_to_a_registered_claim", work="REDESIGN")
        else:
            note("measurement_validity", "adequate", ref=str(probes[-1].get("evidence_id") or ""))

    # -- mechanism support ---------------------------------------------------
    mechanised = [row for row in probes if row.get("mechanism_validated")]
    positive = [row for row in probes if row.get("outcome") == "supports"]
    if not probes:
        force("mechanism_support", "missing", reason="no_experiment", work="PROBE")
    elif not mechanised and positive:
        force("mechanism_support", "weak",
              reason="positive_result_without_mechanism_validation", work="REDESIGN")
    elif not mechanised:
        # Nothing positive to attribute. Naming a positive result here sent a
        # null into REDESIGN to repair a mechanism claim nobody had made.
        force("mechanism_support", "missing",
              reason="no_positive_result_to_attribute_to_a_mechanism", work="PROBE")
    else:
        note("mechanism_support", "adequate", ref=str(mechanised[-1].get("evidence_id") or ""))

    # -- robustness ----------------------------------------------------------
    if not probes:
        force("robustness", "missing", reason="no_experiment", work="PROBE")
    else:
        readings = [(row, check_status(row, PLACEBO)) for row in probes]
        separated = [row for row, status in readings if status == PASS]
        if any(status == FAIL for _, status in readings):
            force("robustness", "missing",
                  reason="effect_not_separable_from_placebo_worlds", work="REDESIGN")
        elif any(status == UNDEFINED for _, status in readings):
            force("robustness", "weak",
                  reason="directional_result_without_a_placebo_comparison", work="PROBE")
        elif separated:
            note("robustness", "adequate", ref=str(separated[-1].get("evidence_id") or ""))
        else:
            force("robustness", "missing",
                  reason="no_directional_result_to_set_against_placebo_worlds", work="PROBE")

    # -- replication / generalization ---------------------------------------
    replications = [row for row in probes if row.get("replication_of")]
    transferred = [row for row in replications if row.get("transfer_ok")]
    independent = [row for row in transferred if row.get("independent_replication")
                   and row.get('outcome') == 'supports' and corroborates(row)]
    if not probes:
        force("replication", "missing", reason="nothing_to_replicate", work="PROBE")
    elif not replications:
        heldout = [row for row in getattr(state, "world_versions", []) or []
                   if row.get("research_role") == "heldout"]
        force("replication", "missing",
              reason=("no_heldout_world_can_host_the_registered_experiment"
                      if heldout and not _replication_possible(state)
                      else "no_heldout_replication_attempted"),
              work="ACQUIRE" if heldout and not _replication_possible(state) else "VERIFY")
    elif not transferred:
        force("replication", "weak", reason="replication_attempted_but_did_not_transfer",
              work="REDESIGN")
    else:
        note("replication", "adequate", ref=str(transferred[-1].get("evidence_id") or ""))
    if independent:
        note("generalization", "adequate", ref=str(independent[-1].get("evidence_id") or ""))
    elif transferred:
        note("generalization", "partial")
    elif probes:
        force("generalization", "missing",
              reason="single_world_result_does_not_generalize", work="VERIFY")

    # -- central claim support ----------------------------------------------
    aligned_ids = set()
    questions = (
        list(getattr(state, "open_questions", []) or [])
        + list(getattr(state, "resolved_questions", []) or [])
        + list(getattr(state, "partial_questions", []) or [])
        + list(getattr(state, "blocked_questions", []) or [])
    )
    for question in questions:
        if alignment_of_question(contract, question, known=questions).status in {"aligned", "instrumental"}:
            aligned_ids.add(str(question.get("id") or ""))
    live_theories = [row for row in theories if row.get("status") != "retired"]
    # Prose has no structure this project will trust. A caller who can state the
    # mission as a registered comparison declares it, and the kernel structures
    # its own rendering of that declaration — a statement checked, not an
    # inference about what the English meant.
    root_structure = (structured_claim(contract.mission)
                      or structured_from_execution(contract.mission_claim))
    mission_relations: dict[str, dict[str, Any]] = {}

    def root_relation(theory: Mapping[str, Any]) -> dict[str, Any]:
        """How this hypothesis stands to the mission's own proposition.

        Identical text is the one relation that needs no rule. Past it only the
        kernel's structured comparison speaks, and it says one of three things.
        The relation that used to be missing is ``incompatible``: a hypothesis
        registered as the opposite comparison on the same world, observable and
        conditions is *on* the root question — confirming it answers that
        question in the negative. Everything else stays unknown, and unknown is
        never softened by repetition or by a worker asserting a link.
        """
        tid = str(theory.get("id") or "")
        if tid in mission_relations:
            return mission_relations[tid]
        text = str(theory.get("claim") or theory.get("text") or "").strip()
        if text and text == str(contract.mission).strip():
            row = {"relation": EQUIVALENT, "basis": "exact_declared_mission_claim",
                   "rule_version": RELATION_RULE_VERSION, "premises": []}
        elif text and root_structure is not None:
            # Compare against the structure in hand, which may have come from a
            # declaration rather than from the mission text. Re-deriving it from
            # `contract.mission` here would discard the declaration and send a
            # prose mission straight back to `unknown`.
            derived = relate(structured_claim(text), root_structure)
            row = {"relation": derived["relation"],
                   "basis": "derived_structured_claim_relation",
                   "rule_version": derived["rule_version"],
                   "premises": list(derived["premises"]), "reason": derived["reason"]}
        else:
            row = {"relation": RELATION_UNKNOWN,
                   "basis": "no_trusted_structure_for_the_root_question",
                   "rule_version": RELATION_RULE_VERSION, "premises": []}
        mission_relations[tid] = row
        return row

    def answers_root(theory: Mapping[str, Any]) -> bool:
        if contract.research_mode != "open_research" and theory.get("role") != "operational_claim":
            return True
        # Question membership is not an entailment: a measured local operation
        # cannot answer a broader root merely because the worker attached Q1.
        return root_relation(theory)["relation"] in {EQUIVALENT, INCOMPATIBLE}

    central_theories = [
        row for row in live_theories
        if row.get("role") != "competing_explanation" and answers_root(row)
        and (not aligned_ids or str(row.get("question_id") or "") in aligned_ids)
    ]
    # Standing on the root question is not the same as standing behind it. A
    # hypothesis whose relation to the mission is ``incompatible`` addresses the
    # question and can never raise support for the mission's own proposition.
    supporting_ids = {str(t.get("id") or "") for t in central_theories
                      if mission_relations.get(str(t.get("id") or ""), {}).get(
                          "relation") != INCOMPATIBLE}
    central_verified = [tid for tid in verified if tid in supporting_ids]
    central_ids = set(supporting_ids)
    # Which claims can *answer* the root question, and how their outcome reads
    # when the question is asked. An equivalent claim reads directly; an
    # incompatible one reads inverted, because both registered margins are
    # positive and a statistic cannot clear both.
    answer_relation: dict[str, str] = {}
    for row in central_theories:
        tid = str(row.get("id") or "")
        text = str(row.get("claim") or row.get("text") or "").strip()
        if text and text == contract.root_question.strip():
            answer_relation[tid] = EQUIVALENT
        elif root_structure is not None:
            derived = root_relation(row)["relation"]
            if derived in {EQUIVALENT, INCOMPATIBLE}:
                answer_relation[tid] = derived
    def answerable(r: Mapping[str, Any]) -> bool:
        return not (r.get('study_stage') in {'pilot', 'calibration', 'scientific_pilot'}
                    or not corroborates(r) or r.get('outcome') not in {'supports', 'weakens'}
                    or (isinstance(r.get('inference_link'), Mapping)
                        and not r['inference_link'].get('inherits_to_scientific_hypothesis')))

    # A registered multi-setting study is answered by its aggregate, never by
    # whichever of its settings came back first. Until every registered setting
    # has an admitted reading the study concludes nothing, and the settings that
    # did run cannot be read as the answer between them.
    studies = {}
    for r in probes:
        tid = str(r.get('theory_id') or '')
        execution = ((r.get('registration') or {}).get('execution')
                     if isinstance(r.get('registration'), Mapping) else None)
        if tid in answer_relation and study_of(execution) is not None and tid not in studies:
            studies[tid] = (execution, study_aggregate(
                probes, execution, claim_id=str(r.get('discovery_idea_id') or tid),
                registration_digest=str(r.get('registration_digest') or ''),
                study_stage=str(r.get('study_stage') or 'confirmatory')))
    answers, readings, study_views = [], [], {}
    for tid, (execution, summary) in studies.items():
        study_views[tid] = summary
        members = [r for r in probes if str(r.get('evidence_id') or '')
                   in set(summary.get('evidence_ids') or ())]
        if (summary.get('status') != 'complete' or not summary.get('verdict')
                or not members or not all(answerable(r) for r in members)):
            continue
        reading = invert_outcome(summary['verdict'], answer_relation[tid])
        if not reading:
            continue
        answers.extend(members)
        readings.append(reading)
    for r in probes:
        tid = str(r.get('theory_id') or '')
        relation = answer_relation.get(tid)
        if not relation or tid in studies or not answerable(r):
            continue
        reading = invert_outcome(r['outcome'], relation)
        if not reading:
            continue
        answers.append(r)
        readings.append(reading)
    relations = set(readings)
    # An answer is adequate when a confirmatory measurement carried it and no
    # recorded sample gap undercuts that measurement. Adequacy is a property of
    # the answer, not of its sign: a refutation established this way is as
    # complete an answer as a confirmation, and still leaves the original
    # hypothesis refuted rather than supported.
    settled = [r for r in answers
               if not any(c.get('check') == 'sample' and c.get('status') in {'fail', 'undefined'}
                          for c in (r.get('validity') or {}).get('checks', []))]
    adequate = bool(settled) and len(relations) == 1 and (
        relations != {'supports'} or bool(central_verified))
    # A contract that names a positive performance target is not satisfied by
    # learning the target is not met. The answer is still an answer; it explains
    # the shortfall instead of completing the objective.
    positive_required = bool(
        contract.research_mode == 'metric_optimization'
        or any((contract.success_metric or {}).get(key) is not None
               for key in ('target', 'threshold', 'must_improve')))
    # What the project did establish, when it could not be promoted. A prose
    # mission leaves `status: unresolved`, and four live runs made that read as
    # "nothing happened" — they had each confirmed a registered comparison. The
    # findings are reported in their own scope and are never promoted by being
    # listed here.
    def _settled(row: Mapping[str, Any]) -> bool:
        return (str(row.get('study_stage') or '') in {'confirmatory', 'replication'}
                and corroborates(row) and row.get('outcome') in {'supports', 'weakens'})

    answered_by = {str(r.get('evidence_id') or '') for r in answers} if len(relations) == 1 else set()
    operational_findings = []
    for row in probes:
        if not _settled(row):
            continue
        theory = next((t for t in live_theories
                       if str(t.get('id') or '') == str(row.get('theory_id') or '')), {})
        claim = str(theory.get('claim') or theory.get('text') or '')
        if not claim:
            continue
        operational_findings.append({
            'theory_id': str(row.get('theory_id') or ''),
            'claim': claim,
            'outcome': str(row.get('outcome') or ''),
            'evidence_id': str(row.get('evidence_id') or ''),
            'scope': 'registered_operation_on_attested_world',
            # True only when this very reading is one the answer rests on.
            # Listing a finding never promotes it; this records whether the
            # relation rule already had.
            'promoted_to_mission': str(row.get('evidence_id') or '') in answered_by,
        })
    # The comparison the kernel actually measured, offered back so a caller can
    # decide whether it is their mission. It is a proposal for a person to
    # approve, never a promotion: nothing reads this field to decide anything.
    candidate_mission_claim: dict[str, Any] = {}
    if root_structure is None:
        for row in probes:
            execution = ((row.get('registration') or {}).get('execution')
                         if isinstance(row.get('registration'), Mapping) else None)
            if not _settled(row) or not isinstance(execution, Mapping):
                continue
            proposal = {key: execution.get(key) for key in
                        ('operation', 'parameters', 'metric', 'expected_direction')}
            if structured_from_execution(proposal) is None:
                continue
            candidate_mission_claim = {
                'mission_claim': proposal,
                'measured_by': str(row.get('evidence_id') or ''),
                'status': 'proposed_for_caller_approval',
                'note': ('this is the comparison the project measured, not a reading of '
                         'what the mission text means; declaring it is the caller\'s decision'),
            }
            break

    status = ('supported_within_tested_conditions' if relations == {'supports'} else
              'refuted_within_tested_conditions' if relations == {'weakens'} else
              'conflicting' if len(relations) > 1 else 'unresolved')
    # A scoped answer is separate from support for the originally proposed sign.
    question_answer = {
        'status': status,
        'strength': 'adequate' if adequate else 'partial' if len(relations) == 1 else 'missing',
        'evidence_ids': [r['evidence_id'] for r in answers],
        'scope': 'registered_claim_under_tested_conditions_only',
        'margin_basis': 'the registered decision margin of the answering experiment',
        'relation_rule_version': RELATION_RULE_VERSION,
        'relations': {tid: dict(mission_relations.get(tid) or {}, relation=relation)
                      for tid, relation in sorted(answer_relation.items())},
        'completes_open_research': not (positive_required
                                        and status == 'refuted_within_tested_conditions'),
        'positive_target_required': positive_required,
        # Why there is no answer, when there is none. "Unresolved" on its own
        # sent four live missions off to redesign experiments that were correct:
        # they had measured exactly what was asked and simply had no structured
        # mission to promote it to.
        'unresolved_reason': (
            '' if status != 'unresolved' else
            'mission_has_no_structured_definition' if root_structure is None else
            'no_admitted_measurement_answers_the_mission_claim'),
        'unlock': ('' if status != 'unresolved' or root_structure is not None else
                   'declare the mission as a registered comparison in '
                   'contract.mission_claim, or state it as the registered sentence'),
        'operational_findings': operational_findings,
        'candidate_mission_claim': candidate_mission_claim,
        'studies': {tid: {'status': view.get('status', ''), 'verdict': view.get('verdict', ''),
                          'method': view.get('method', ''), 'scope': view.get('scope', ''),
                          'coverage': view.get('coverage', {})}
                    for tid, view in sorted(study_views.items())},
    }
    central_support = [row for row in probes
                       if str(row.get("theory_id") or "") in central_ids
                       and row.get("outcome") == "supports"
                       and str(row.get("study_stage") or "") not in {
                           "pilot", "calibration", "scientific_pilot"}
                       and (not isinstance(row.get("inference_link"), Mapping)
                            or row["inference_link"].get("inherits_to_scientific_hypothesis"))]
    if central_verified:
        note("central_claim_support", "adequate", ref=central_verified[0])
    elif any(corroborates(row) for row in central_support):
        note("central_claim_support", "partial",
             reason="supported_but_not_promoted_to_verified", work="VERIFY")
    elif central_support:
        # A positive reading that cannot be credited to the claim — its power
        # unknown, a competitor unablated — is not partial support.
        force("central_claim_support", "weak",
              reason="supporting_results_cannot_corroborate", work="REDESIGN")
    elif any(str(row.get("study_stage") or "") not in {
            "pilot", "calibration", "scientific_pilot"} for row in probes):
        force("central_claim_support", "weak",
              reason="no_supporting_evidence_for_a_root_aligned_claim", work="REDESIGN")
    else:
        force("central_claim_support", "missing", reason="no_evidence_for_the_root_question",
              work="PROBE")
    if live_theories and not central_theories:
        # Every hypothesis this project is still pursuing has left the root
        # question. Retiring one direction does not count: that is a project
        # that already changed direction, not one that needs to.
        force("central_claim_support", "missing",
              reason="active_hypotheses_do_not_establish_mission_coverage",
              work="REVIEW")

    # -- reviewer objections override upward-looking projections -------------
    for row in open_reviews(reviews):
        record = ReviewRecord.from_dict(row)
        name = record.dimension if record.dimension in dims else REPAIR_DIMENSION.get(
            record.repair_type, "technical_correctness")
        ceiling = {"fatal": "missing", "major": "weak", "minor": "partial"}[record.severity]
        target = dims[name]
        target.status = cap(target.status, ceiling)
        target.blocking_reasons.append(f"{record.severity}_review:{record.review_id}")
        target.required_next_work.append(_work_for_repair(record.repair_type))
        if record.severity == "fatal":
            central = dims["central_claim_support"]
            central.status = cap(central.status, "missing")
            central.blocking_reasons.append(f"fatal_review:{record.review_id}")

    # -- judgment dimensions: assessed, and only as far as their grounding ---
    # No execution settles novelty or importance, so a recorded judgment is the
    # only thing that can move them. A bare opinion is capped at ``partial``;
    # reaching ``adequate`` requires the judgment to cite literature this state
    # actually verified. An assessment is not an objection and does not need to
    # be open, but an open objection on the same dimension still caps it below.
    citations = {
        str(cite)
        for row in surveys
        for cite in list(row.get("citations") or []) + list(row.get("read_first") or [])
    }
    from .judgment import SOURCE as PANEL_SOURCE, judgment_subject
    subject_digest = str(judgment_subject(state).get("digest") or "")
    for name in JUDGMENT_DIMENSIONS:
        assessments = [
            row for row in reviews
            if ReviewRecord.from_dict(row).dimension == name
            and str(row.get("assessment") or "") in STATUS_RANK
        ]
        # A panel judged a particular subject — these claims, these outcomes,
        # this literature. Its grade governs while that is still the subject and
        # counts for nothing afterwards; and while it governs, a later bare
        # opinion cannot outvote it. Taking the most favourable of all recorded
        # assessments, as this once did, meant every added reviewer could only
        # raise a grade.
        paneled = [row for row in assessments if row.get("source") == PANEL_SOURCE]
        current = [row for row in paneled
                   if str((row.get("panel") or {}).get("subject_digest") or "") == subject_digest]
        stale = bool(paneled) and not current
        assessments = current[-1:] if current else [row for row in assessments if row not in paneled]
        if stale:
            dims[name].required_next_work.append("REVIEW")
            dims[name].blocking_reasons.append(
                "judged_for_an_earlier_subject; the claims, outcomes or literature have changed")
        if assessments:
            best, refs = "unknown", []
            for row in assessments:
                grounded = [c for c in (row.get("supporting_refs") or []) if str(c) in citations]
                ceiling = "adequate" if grounded else "partial"
                proposed = cap(str(row["assessment"]), ceiling)
                if STATUS_RANK[proposed] > STATUS_RANK[best]:
                    best, refs = proposed, grounded
            dims[name].status = best
            dims[name].supporting_refs.extend(str(c) for c in refs)
            if not refs:
                dims[name].blocking_reasons.append(
                    "judgment is not grounded in verified literature; capped at partial")
        elif stale:
            pass
        elif dims[name].status == "unknown" and not any(
            ReviewRecord.from_dict(row).dimension == name for row in reviews
        ):
            dims[name].required_next_work.append("REVIEW")
            dims[name].blocking_reasons.append("no_recorded_judgment_and_no_execution_can_supply_one")
    # Objections outrank assessments: a reviewer cannot rate away their own
    # unresolved complaint.
    for row in open_reviews(reviews):
        record = ReviewRecord.from_dict(row)
        if record.dimension in JUDGMENT_DIMENSIONS:
            ceiling = {"fatal": "missing", "major": "weak", "minor": "partial"}[record.severity]
            dims[record.dimension].status = cap(dims[record.dimension].status, ceiling)

    requirements = requirements_for(contract)
    gaps = [
        name for name, requirement in sorted(requirements.items())
        if not meets(dims[name].status, requirement)
        and not (name == 'central_claim_support'
                 and 'central_question_answered' in contract.termination_requirements
                 and question_answer['completes_open_research']
                 and meets(question_answer['strength'], requirement))
    ]
    remaining_mission_gaps = [
        {"question_id": str(question.get("id") or ""), "gap": str(gap)}
        for question in questions for gap in question.get("remaining_mission_gaps") or []
    ]
    reconsiderations = [str(question.get("id") or "") for question in questions
                        if question.get("question_basis_status") == "reconsideration_required"
                        or question.get("transition_status") == "reconsideration_required"]
    mission_completion = {
        "status": ("answered_within_tested_conditions" if answers and len(relations) == 1
                   and not remaining_mission_gaps and not reconsiderations else "unresolved"),
        "basis": ("exact_declared_mission_claim_only" if root_structure is None
                  else "declared_mission_claim_or_checkable_structured_relation"),
        "semantic_coverage": "unknown",
        "remaining_mission_gaps": remaining_mission_gaps,
        "questions_requiring_reconsideration": reconsiderations,
        "evidence_ids": list(question_answer["evidence_ids"]),
    }
    return {
        "rule": "ordinal-derived-quality-v1",
        "target_quality": contract.target_quality,
        "requirements": requirements,
        "dimensions": {name: dims[name].to_dict() for name in DIMENSIONS},
        "quality_gap": gaps,
        "quality_gap_size": len(gaps),
        "review_debt": review_debt(reviews),
        "question_answer": question_answer,
        "mission_completion": mission_completion,
        "basis": "derived_from_admitted_evidence_and_recorded_reviews_only",
    }


def _work_for_repair(repair_type: str) -> str:
    if repair_type in PIVOT_REPAIRS:
        return "PROJECT_PIVOT"
    if repair_type in REDESIGN_REPAIRS:
        return "REDESIGN"
    if repair_type == "stronger_baseline":
        return "REDESIGN"
    if repair_type == "broader_evaluation":
        return "VERIFY"
    if repair_type == "more_evidence":
        return "PROBE"
    return "SYNTHESIZE"


def quality_gap(state: Any) -> list[str]:
    quality = getattr(state, "quality", None)
    if isinstance(quality, Mapping) and "quality_gap" in quality:
        return [str(item) for item in quality.get("quality_gap") or []]
    return list(compute_quality(state).get("quality_gap") or [])
