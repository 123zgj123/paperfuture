"""Persistent scientific control state. Not evidence, not a WorldID.

Notebook and researchspace.ResearchState are not this object.
This is the only control state the scheduler reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..validity import corroborates, judge_probe_row, sample_floor
from .kernel import refuse_generated_promotion, readonly_view


SCIENTIFIC_STATE_FILE = "SCIENTIFIC_STATE.json"


def _admitted_observation(row: Mapping[str, Any]) -> bool:
    """Provenance and execution gates shared by observations and confirmation."""
    return bool(str(row.get("role") or "") == "ProbeEvidence"
                and row.get("epistemic") == "WORLD" and row.get("attested")
                and not refuse_generated_promotion(row) and not row.get("promotion_refused")
                and row.get("execution_status") == "ran"
                and row.get("experiment_digest") and row.get("world_digest")
                and not row.get("invalidated")
                and not (row.get("verify_count") and (row.get("reproduction_ok") is False
                                                     or row.get("replay_ok") is False)))


def evidence_reliability(row: Mapping[str, Any]) -> float:
    """Compatibility eligibility indicator, never a weight or confidence."""
    return float(_admitted_observation(row) and bool(row.get("promotion_ok"))
                 and corroborates(row))


def refresh_research_views(state: "ScientificState") -> None:
    """Derive scoped categorical belief from still-valid recorded evidence.

    Observations constrain planning without becoming central corroboration.
    Repeated source/freeze identities contribute once; no count, critic score,
    or uncalibrated number is presented as scientific confidence.
    """
    limits = dict(getattr(state, "contract", None) or {}).get("resource_limits")
    floor = sample_floor(limits if isinstance(limits, Mapping) else None)
    for row in state.evidence_records:
        if str(row.get("role") or "") == "ProbeEvidence":
            judge_probe_row(row, min_sample=floor)
    beliefs: dict[str, dict[str, Any]] = {}
    verified: dict[str, dict[str, Any]] = {}
    for theory in state.theories:
        tid = str(theory.get("id") or "")
        if not tid:
            continue
        belief: dict[str, Any] = {
            "rule": "scoped-evidence-v2", "prior_confidence": None,
            "posterior_confidence": None, "status": "insufficient",
            "supporting_evidence": [], "contradicting_evidence": [],
            "observation_evidence": [], "observation_status": "none",
            "unresolved_gaps": [], "scope": [],
            "competing_theories": list(theory.get("competing") or theory.get("competing_explanations") or []),
            "unresolved_assumptions": list(theory.get("assumptions") or []),
            "update_provenance": [],
        }
        seen: set[tuple[str, str]] = set()
        rows = [r for r in state.evidence_records if r.get("theory_id") == tid]
        observed_directions: set[str] = set()
        invalidated = False

        def gap(kind: str, reason: str, eid: str = "") -> None:
            item = {"id": kind, "reason": reason, "evidence_id": eid}
            if item not in belief["unresolved_gaps"]:
                belief["unresolved_gaps"].append(item)

        # A completed confirmation supersedes a limited reading of the same
        # source/freeze identity, even when the pilot entered the log first.
        for row in sorted(rows, key=lambda item: not evidence_reliability(item)):
            eid = str(row.get("evidence_id") or "")
            if not eid or row.get("epistemic") != "WORLD" or refuse_generated_promotion(row):
                continue
            if row.get("invalidated"):
                invalidated = True
                gap("invalidated", str(row.get("invalidation_reason") or "evidence was invalidated"), eid)
                continue
            validity = row.get("validity") or {}
            for check in validity.get("checks") or []:
                if check.get("status") in {"fail", "undefined"}:
                    gap(str(check["check"]), str(check.get("reason") or "check was not satisfied"), eid)
            if not _admitted_observation(row):
                gap("provenance", "an attested, identified, successfully executed observation is required", eid)
                continue
            if not validity.get("measurement_valid"):
                continue
            identity = (str(row.get("world_digest")), str(row.get("experiment_digest")))
            if identity in seen:
                continue
            seen.add(identity)
            outcome = row.get("outcome") or row.get("verdict")
            if evidence_reliability(row):
                bucket = "supporting_evidence" if outcome == "supports" else "contradicting_evidence"
                belief[bucket].append(eid)
                contribution = "scoped_support" if outcome == "supports" else "scoped_refutation"
            else:
                belief["observation_evidence"].append(eid)
                observed_directions.add(str(outcome))
                contribution = "limited_observation"
                if row.get("study_stage") in {"pilot", "calibration", "scientific_pilot"}:
                    gap("study_stage", "pilot or calibration measurements require a confirmatory study", eid)
                if outcome == "uninformative":
                    gap("discrimination", "the registered comparison did not discriminate the claim", eid)
                if not row.get("promotion_ok"):
                    gap("confirmation", "the observation is not admitted for central corroboration", eid)
            belief["scope"].append({
                "evidence_id": eid, "world_id": row.get("world_id", ""),
                "world_digest": identity[0], "experiment_digest": identity[1],
                "supports_scope": list(row.get("supports_scope") or []),
                "prohibited_inferences": list(row.get("prohibited_inferences") or []),
            })
            belief["update_provenance"].append({
                "evidence_id": eid, "event_id": row.get("scientific_event_id", ""),
                "verification_event_id": row.get("verification_event_id", ""),
                "contribution": contribution, "outcome": outcome,
                "world_digest": identity[0], "experiment_digest": identity[1],
            })
        supporting, contradicting = belief["supporting_evidence"], belief["contradicting_evidence"]
        if supporting and contradicting:
            belief["status"] = "conflicting"
            gap("conflict", "valid confirmatory results point in opposing directions")
        elif contradicting:
            belief["status"] = "refuted"
        elif supporting:
            belief["status"] = "supported"
        elif belief["observation_evidence"]:
            belief["status"] = ("insufficient" if any(g["id"] == "sample" for g in belief["unresolved_gaps"])
                                else "inconclusive")
        elif invalidated:
            belief["status"] = "invalidated"
        else:
            gap("confirmation", "no valid confirmatory evidence is available")
        if {"supports", "weakens"}.issubset(observed_directions):
            belief["observation_status"] = "conflicting"
        elif "weakens" in observed_directions:
            belief["observation_status"] = "directional_refutation_observed"
        elif "supports" in observed_directions:
            belief["observation_status"] = "directional_support_observed"
        elif observed_directions:
            belief["observation_status"] = "inconclusive"
        elif invalidated:
            belief["observation_status"] = "invalidated"
        theory["prior_confidence"] = None
        theory["posterior_confidence"] = None
        theory["belief_state_id"] = tid
        # Project retirement is intentional; every evidential status is derived
        # afresh, including clearing support after its evidence is invalidated.
        if str(theory.get("status") or "") != "retired":
            theory["status"] = belief["status"]
        beliefs[tid] = belief
        if theory.get('status') == 'retired':
            continue
        # A corroborated execution is not automatically a verified mechanism.
        for row in rows:
            if not (evidence_reliability(row) and row.get("outcome") == "supports"
                    and all(row.get(k) for k in ("literature_checked", "reproduction_ok", "identity_ok",
                                                 "executable_ok", "mechanism_validated"))):
                continue
            replication = next((r for r in rows if r.get("replication_of") == row.get("evidence_id")
                and r.get("world_digest") != row.get("world_digest") and evidence_reliability(r)
                and r.get("outcome") == "supports" and r.get("identity_ok") and r.get("reproduction_ok")), None)
            if row.get("replication_required", True) and replication is None:
                continue
            if belief["contradicting_evidence"]:
                continue  # unresolved contradictions keep the mechanism speculative
            verified[tid] = {"theory_id": tid, "evidence_id": row["evidence_id"],
                             "replication_evidence_id": replication["evidence_id"] if replication else "",
                             "claim_strength": "verified_within_tested_conditions",
                             "world_id": row.get("world_id"), "world_digest": row.get("world_digest")}
            break
    state.belief_states = beliefs
    state.verified_research_state = verified
    retired = {t.get('id') for t in state.theories if t.get('status') in {'retired', 'refuted'}}
    state.speculative_frontier = [tid for tid in beliefs if tid not in verified and tid not in retired]
    refresh_contradictions(state)
    from .discovery import sync_research_frontier
    sync_research_frontier(state)


def refresh_contradictions(state: "ScientificState") -> None:
    """Derive contradictions from admitted evidence that disagrees with itself.

    ``state.contradictions`` had exactly two writers: a ``ContradictionDetected``
    event no code ever emitted, and a branch-merge helper with no live caller.
    Everything that consumed it — verification debt, VERIFY eligibility, the
    readiness gate, the belief refresh's own "unresolved contradictions keep the
    mechanism speculative" rule — was reading a list that could never fill. A
    listed capability that cannot fire is worse than an absent one, because the
    gate it guards reads as satisfied.

    A contradiction here is a fact, not a judgment: two independently admitted
    WORLD rows about the same hypothesis, on different experiment identities,
    pointing opposite ways. Two runs of the same experiment on the same freeze
    are one observation and are not in tension with themselves.

    Rows reported explicitly through the event remain; derived rows are
    recomputed from canonical evidence on every replay, so the two never fight.
    """
    retired = {r.get('id') for r in state.theories if r.get('status') == 'retired'}
    reported = [dict(row) for row in state.contradictions
                if str(row.get("source") or "") == "reported" and row.get('theory_id') not in retired]
    derived: list[dict[str, Any]] = []
    for theory in state.theories:
        tid = str(theory.get("id") or "")
        if tid in retired:
            continue
        belief = state.belief_states.get(tid) or {}
        supporting = list(belief.get("supporting_evidence") or [])
        contradicting = list(belief.get("contradicting_evidence") or [])
        if not (supporting and contradicting):
            continue
        by_id = {str(row.get("evidence_id") or ""): row for row in state.evidence_records}
        for positive in supporting:
            for negative in contradicting:
                left, right = by_id.get(positive, {}), by_id.get(negative, {})
                if (left.get("world_digest"), left.get("experiment_digest")) == (
                        right.get("world_digest"), right.get("experiment_digest")):
                    continue  # the same experiment cannot contradict itself
                derived.append({
                    "id": f"contradiction:{tid}:{positive}:{negative}",
                    "theory_id": tid,
                    "question_id": str(left.get("question_id") or right.get("question_id") or ""),
                    "supporting_evidence_id": positive,
                    "contradicting_evidence_id": negative,
                    "worlds": sorted({str(left.get("world_id") or ""),
                                      str(right.get("world_id") or "")}),
                    "source": "derived",
                    "basis": "two_admitted_world_results_on_the_same_hypothesis_disagree",
                    "epistemic": "WORLD",
                })
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for row in reported + derived:
        key = str(row.get("id") or row.get("evidence_id") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(row)
    state.contradictions = merged


def _copy_rows(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows or []:
        if isinstance(item, Mapping):
            out.append(dict(item))
        elif str(item).strip():
            out.append({"text": str(item).strip()})
    return out


def _copy_text(rows: Any) -> list[str]:
    seen: list[str] = []
    for item in rows or []:
        if isinstance(item, Mapping):
            text = str(item.get("text") or item.get("id") or item.get("target") or "").strip()
        else:
            text = str(item).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


@dataclass
class ScientificState:
    """Control state for the open-world scheduler.

    GENERATED rows may be listed. Listing is not attestation.
    Mutation of these fields in production goes through the reducer.
    """

    goal: str = ""
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    resolved_questions: list[dict[str, Any]] = field(default_factory=list)
    partial_questions: list[dict[str, Any]] = field(default_factory=list)
    blocked_questions: list[dict[str, Any]] = field(default_factory=list)
    theories: list[dict[str, Any]] = field(default_factory=list)
    competing_explanations: list[str] = field(default_factory=list)
    world_versions: list[dict[str, Any]] = field(default_factory=list)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    accepted_ideas: list[dict[str, Any]] = field(default_factory=list)
    rejected_ideas: list[dict[str, Any]] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    available_capabilities: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    failed_designs: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    archive_ids: list[str] = field(default_factory=list)
    active_lineages: list[str] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    harness_version: str = "H0"
    world_id: str = ""
    ticks: int = 0
    last_progress_tick: int = 0
    # Operational idleness and research stagnation are different clocks.
    last_research_progress_tick: int = 0
    event_seq: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    frontier: dict[str, Any] = field(default_factory=dict)
    debt: dict[str, int] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    belief_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    speculative_frontier: list[str] = field(default_factory=list)
    verified_research_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Project-level control state. A goal string survives one action; these
    # survive a hundred. All of them are event-sourced and replay-derived.
    contract: dict[str, Any] = field(default_factory=dict)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    design_revisions: list[dict[str, Any]] = field(default_factory=list)
    retired_lineages: list[dict[str, Any]] = field(default_factory=list)
    decision_journal: list[dict[str, Any]] = field(default_factory=list)
    project_epoch: int = 0
    research_branches: list[dict[str, Any]] = field(default_factory=list)
    research_ledger: list[dict[str, Any]] = field(default_factory=list)
    decision_outcomes: list[dict[str, Any]] = field(default_factory=list)
    reasoning_mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "open_questions": list(self.open_questions),
            "resolved_questions": list(self.resolved_questions),
            "partial_questions": list(self.partial_questions),
            "blocked_questions": list(self.blocked_questions),
            "theories": list(self.theories),
            "competing_explanations": list(self.competing_explanations),
            "world_versions": list(self.world_versions),
            "evidence_records": list(self.evidence_records),
            "accepted_ideas": list(self.accepted_ideas),
            "rejected_ideas": list(self.rejected_ideas),
            "missing_capabilities": list(self.missing_capabilities),
            "available_capabilities": list(self.available_capabilities),
            "artifacts": list(self.artifacts),
            "failed_designs": list(self.failed_designs),
            "anomalies": list(self.anomalies),
            "archive_ids": list(self.archive_ids),
            "active_lineages": list(self.active_lineages),
            "contradictions": list(self.contradictions),
            "harness_version": self.harness_version,
            "world_id": self.world_id,
            "ticks": self.ticks,
            "last_progress_tick": self.last_progress_tick,
            "last_research_progress_tick": self.last_research_progress_tick,
            "event_seq": self.event_seq,
            "metrics": dict(self.metrics),
            "frontier": dict(self.frontier),
            "debt": dict(self.debt),
            "graph": dict(self.graph),
            "belief_states": dict(self.belief_states),
            "speculative_frontier": list(self.speculative_frontier),
            "verified_research_state": dict(self.verified_research_state),
            "contract": dict(self.contract),
            "reviews": list(self.reviews),
            "quality": dict(self.quality),
            "resources": dict(self.resources),
            "objective": dict(self.objective),
            "design_revisions": list(self.design_revisions),
            "retired_lineages": list(self.retired_lineages),
            "decision_journal": list(self.decision_journal),
            "project_epoch": self.project_epoch,
            "research_branches": list(self.research_branches),
            "research_ledger": list(self.research_ledger),
            "decision_outcomes": list(self.decision_outcomes),
            "reasoning_mode": self.reasoning_mode,
        }

    def view(self) -> Mapping[str, Any]:
        """Read-only snapshot. Executors must not write through this."""
        return readonly_view(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ScientificState":
        row = dict(payload or {})
        return cls(
            goal=str(row.get("goal") or ""),
            open_questions=_copy_rows(row.get("open_questions")),
            resolved_questions=_copy_rows(row.get("resolved_questions")),
            partial_questions=_copy_rows(row.get("partial_questions")),
            blocked_questions=_copy_rows(row.get("blocked_questions")),
            theories=_copy_rows(row.get("theories")),
            competing_explanations=_copy_text(row.get("competing_explanations")),
            world_versions=_copy_rows(row.get("world_versions")),
            evidence_records=_copy_rows(row.get("evidence_records")),
            accepted_ideas=_copy_rows(row.get("accepted_ideas")),
            rejected_ideas=_copy_rows(row.get("rejected_ideas")),
            missing_capabilities=_copy_text(row.get("missing_capabilities")),
            available_capabilities=_copy_text(row.get("available_capabilities")),
            artifacts=_copy_rows(row.get("artifacts")),
            failed_designs=_copy_rows(row.get("failed_designs")),
            anomalies=_copy_text(row.get("anomalies")),
            archive_ids=_copy_text(row.get("archive_ids")),
            active_lineages=_copy_text(row.get("active_lineages")),
            contradictions=_copy_rows(row.get("contradictions")),
            harness_version=str(row.get("harness_version") or "H0"),
            world_id=str(row.get("world_id") or ""),
            ticks=int(row.get("ticks") or 0),
            last_progress_tick=int(row.get("last_progress_tick") or 0),
            last_research_progress_tick=int(row.get("last_research_progress_tick") or 0),
            event_seq=int(row.get("event_seq") or 0),
            metrics=dict(row.get("metrics") or {}),
            frontier=dict(row.get("frontier") or {}),
            debt={str(k): int(v) for k, v in dict(row.get("debt") or {}).items()},
            graph=dict(row.get("graph") or {}),
            belief_states=dict(row.get("belief_states") or {}),
            speculative_frontier=_copy_text(row.get("speculative_frontier")),
            verified_research_state=dict(row.get("verified_research_state") or {}),
            contract=dict(row.get("contract") or {}),
            reviews=_copy_rows(row.get("reviews")),
            quality=dict(row.get("quality") or {}),
            resources=dict(row.get("resources") or {}),
            objective=dict(row.get("objective") or {}),
            design_revisions=_copy_rows(row.get("design_revisions")),
            retired_lineages=_copy_rows(row.get("retired_lineages")),
            decision_journal=_copy_rows(row.get("decision_journal")),
            project_epoch=int(row.get("project_epoch") or 0),
            research_branches=_copy_rows(row.get("research_branches")),
            research_ledger=_copy_rows(row.get("research_ledger")),
            decision_outcomes=_copy_rows(row.get("decision_outcomes")),
            reasoning_mode=str(row.get("reasoning_mode") or ""),
        )

    def add_question(
        self,
        *,
        question_id: str,
        text: str,
        required_observable: str = "",
        required_capability: str = "",
        lineage_id: str = "",
        epistemic: str = "GENERATED",
    ) -> None:
        """Test / bootstrap helper. Production writes go through QuestionCreated."""
        row = {
            "id": question_id,
            "text": text,
            "required_observable": required_observable,
            "required_capability": required_capability,
            "lineage_id": lineage_id,
            "epistemic": epistemic or "GENERATED",
            "status": "open",
        }
        if refuse_generated_promotion(row):
            row["epistemic"] = "GENERATED"
        if not any(item.get("id") == question_id for item in self.open_questions):
            self.open_questions.append(row)
        if lineage_id and lineage_id not in self.active_lineages:
            self.active_lineages.append(lineage_id)

    def note_missing(self, name: str) -> None:
        text = str(name or "").strip()
        if text and text not in self.missing_capabilities:
            self.missing_capabilities.append(text)
        if text in self.available_capabilities:
            self.available_capabilities.remove(text)

    def note_available(self, name: str) -> None:
        text = str(name or "").strip()
        if text and text not in self.available_capabilities:
            self.available_capabilities.append(text)
        if text in self.missing_capabilities:
            self.missing_capabilities.remove(text)

    def record_evidence(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        if refuse_generated_promotion(payload) and str(payload.get("epistemic") or "") == "WORLD":
            payload["epistemic"] = "GENERATED"
            payload["promotion_refused"] = refuse_generated_promotion(payload)
        self.evidence_records.append(payload)
        return payload

    def resolve_question(self, question_id: str, *, evidence_id: str = "", partial: bool = False) -> None:
        kept: list[dict[str, Any]] = []
        moved = None
        for row in self.open_questions:
            if row.get("id") == question_id:
                moved = dict(row)
            else:
                kept.append(row)
        self.open_questions = kept
        if moved is None:
            for row in self.blocked_questions:
                if row.get("id") == question_id:
                    moved = dict(row)
                    self.blocked_questions = [
                        item for item in self.blocked_questions if item.get("id") != question_id
                    ]
                    break
        if moved is None:
            return
        moved["status"] = "partial" if partial else "resolved"
        if evidence_id:
            moved["evidence_id"] = evidence_id
        dest = self.partial_questions if partial else self.resolved_questions
        dest.append(moved)

    def block_question(self, question_id: str, *, reason: str = "") -> None:
        kept: list[dict[str, Any]] = []
        moved = None
        for row in self.open_questions:
            if row.get("id") == question_id:
                moved = dict(row)
            else:
                kept.append(row)
        self.open_questions = kept
        if moved is None:
            return
        moved["status"] = "blocked"
        if reason:
            moved["blocked_reason"] = reason
        if not any(item.get("id") == question_id for item in self.blocked_questions):
            self.blocked_questions.append(moved)

    def reopen_question(self, question_id: str) -> None:
        kept: list[dict[str, Any]] = []
        moved = None
        for row in self.blocked_questions:
            if row.get("id") == question_id:
                moved = dict(row)
            else:
                kept.append(row)
        self.blocked_questions = kept
        if moved is None:
            return
        moved["status"] = "open"
        if not any(item.get("id") == question_id for item in self.open_questions):
            self.open_questions.append(moved)


def load_scientific_state(workspace: Path | None) -> ScientificState:
    if workspace is None:
        return ScientificState()
    path = Path(workspace) / SCIENTIFIC_STATE_FILE
    if not path.is_file():
        return ScientificState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ScientificState()
    if not isinstance(payload, dict):
        return ScientificState()
    return ScientificState.from_dict(payload)


def save_scientific_state(workspace: Path | None, state: ScientificState) -> Path | None:
    from .events import write_file_atomically

    if workspace is None:
        return None
    path = Path(workspace) / SCIENTIFIC_STATE_FILE
    write_file_atomically(path, json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n")
    return path


def state_is_empty(state: ScientificState) -> bool:
    return not (
        state.goal
        or state.open_questions
        or state.resolved_questions
        or state.evidence_records
        or state.world_id
        or state.theories
        or state.contract
    )


def sync_from_notebook(state: ScientificState, notebook: Any) -> ScientificState:
    """Bootstrap only. Notebook is not an authority once ScientificState exists."""
    if notebook is None:
        return state
    for text in getattr(notebook, "unresolved_questions", ()) or []:
        if not any(row.get("text") == text for row in state.open_questions):
            state.add_question(
                question_id=f"q-{len(state.open_questions)+1}",
                text=str(text),
                epistemic="GENERATED",
            )
    for gap in getattr(notebook, "missing_capabilities", ()) or []:
        state.note_missing(str(gap))
    for row in getattr(notebook, "world_versions", ()) or []:
        if isinstance(row, Mapping) and row not in state.world_versions:
            state.world_versions.append(dict(row))
            if not state.world_id:
                state.world_id = str(row.get("world_id") or "")
    for row in getattr(notebook, "confirmed_evidence", ()) or []:
        if isinstance(row, Mapping):
            stamped = dict(row)
            stamped.setdefault("epistemic", stamped.get("epistemic") or "GENERATED")
            if stamped not in state.evidence_records:
                state.record_evidence(stamped)
    for text in getattr(notebook, "open_mechanisms", ()) or []:
        if not any(row.get("text") == text for row in state.theories):
            state.theories.append({"text": str(text), "epistemic": "GENERATED"})
    for row in getattr(notebook, "failed_designs", ()) or []:
        if isinstance(row, Mapping) and row not in state.failed_designs:
            state.failed_designs.append(dict(row))
    for lineage in getattr(notebook, "ideas", ()) or []:
        if isinstance(lineage, Mapping):
            lid = str(lineage.get("lineage_id") or "")
            if lid and lid not in state.active_lineages:
                state.active_lineages.append(lid)
    return state


def notebook_projection(state: ScientificState) -> dict[str, Any]:
    """ScientificState → prompt projection. Not an independent truth."""
    return {
        "confirmed_evidence": [
            dict(row)
            for row in state.evidence_records
            if str(row.get("epistemic") or "") == "WORLD"
        ],
        "ruled_out": list(state.anomalies),
        "unresolved_questions": [
            str(row.get("text") or row.get("id") or "") for row in state.open_questions
        ],
        "missing_capabilities": list(state.missing_capabilities),
        "world_versions": list(state.world_versions),
        "failed_designs": list(state.failed_designs),
        "open_mechanisms": [
            str(row.get("text") or row.get("id") or "")
            for row in state.theories
            if str(row.get("epistemic") or "") != "WORLD"
        ],
        "next_actions": list((state.frontier or {}).get("current_frontier_nodes") or []),
        "ideas": list(state.accepted_ideas),
        "projection_of": "ScientificState",
    }
