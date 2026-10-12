"""Harness proposals and fixture-scoped checks for the built-in snapshot helper.

These offline checks are not independent mission credit. The live research
controller records proposals without installing them.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .capability import (
    Capability,
    CapabilityRegistry,
    SNAPSHOT_CAPABILITY,
    run_capability,
)
from .env import HarnessArtifact, HarnessVersion, admit_artifact
from .kernel import (
    KernelViolation,
    assert_mutable_module,
    assert_patch_files,
)


SCIENTIFIC_FAILURE = "SCIENTIFIC_FAILURE"
HARNESS_FAILURE = "HARNESS_FAILURE"
ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
TRANSIENT_EXECUTION_FAILURE = "TRANSIENT_EXECUTION_FAILURE"

PATHOLOGY_FILE = "HARNESS_PATHOLOGY.json"
PATHOLOGY_FLOOR = 3


@dataclass
class Pathology:
    where: str
    why: str
    count: int = 1
    examples: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.where}×{self.why}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "where": self.where,
            "why": self.why,
            "count": self.count,
            "examples": list(self.examples),
            "key": self.key,
        }


@dataclass
class CandidatePatch:
    module: str
    pathology: str
    artifact: HarnessArtifact
    capability: Capability | None = None
    files: tuple[str, ...] = ()
    justification: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "pathology": self.pathology,
            "artifact": self.artifact.to_dict(),
            "capability": self.capability.to_dict() if self.capability else None,
            "files": list(self.files),
            "justification": self.justification,
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass
class CreditReport:
    compile_ok: bool
    failing_replay: bool
    nearby_ok: bool
    regression_ok: bool
    previous_ok: bool
    sealed_gain: bool | None
    cost_ok: bool
    general: bool
    reasons: tuple[str, ...] = ()
    paired_gain: float = 0.0
    artifact_enabled_delta: float = 0.0
    h0_score: float = 0.0
    h1_score: float = 0.0
    h1_disabled_score: float = 0.0
    candidate_digest: str = ""
    evaluation_scope: str = "unverified"

    @property
    def admit(self) -> bool:
        base = (
            self.compile_ok
            and self.failing_replay
            and self.nearby_ok
            and self.regression_ok
            and self.previous_ok
            and self.cost_ok
        )
        if not base:
            return False
        if self.sealed_gain is False:
            return False
        return (bool(self.candidate_digest)
                and self.evaluation_scope == "validator_snapshot_fixtures_v1"
                and all(math.isfinite(value) and 0 <= value <= 1 for value in
                        (self.h0_score, self.h1_score, self.h1_disabled_score))
                and math.isfinite(self.paired_gain) and self.paired_gain > 0
                and math.isfinite(self.artifact_enabled_delta) and self.artifact_enabled_delta > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_ok": self.compile_ok,
            "failing_replay": self.failing_replay,
            "nearby_ok": self.nearby_ok,
            "regression_ok": self.regression_ok,
            "previous_ok": self.previous_ok,
            "sealed_gain": self.sealed_gain,
            "cost_ok": self.cost_ok,
            "general": self.general,
            "admit": self.admit,
            "reasons": list(self.reasons),
            "paired_gain": self.paired_gain,
            "artifact_enabled_delta": self.artifact_enabled_delta,
            "h0_score": self.h0_score,
            "h1_score": self.h1_score,
            "h1_disabled_score": self.h1_disabled_score,
            "candidate_digest": self.candidate_digest,
            "evaluation_scope": self.evaluation_scope,
        }


def classify_failure(event: Mapping[str, Any] | None) -> str:
    row = dict(event or {})
    kind = str(row.get("failure_class") or row.get("class") or "").strip()
    known = {
        SCIENTIFIC_FAILURE,
        HARNESS_FAILURE,
        ENVIRONMENT_FAILURE,
        TRANSIENT_EXECUTION_FAILURE,
    }
    if kind in known:
        return kind
    if row.get("transient") or row.get("timeout") or "unavailable" in str(row.get("why") or ""):
        return TRANSIENT_EXECUTION_FAILURE
    if row.get("environment") or row.get("gpu") or "gpu" in str(row.get("why") or ""):
        return ENVIRONMENT_FAILURE
    if row.get("missing_artifact") or row.get("scientific_gap") or row.get("no_data"):
        return SCIENTIFIC_FAILURE
    if row.get("hypothesis_false") or row.get("verdict") in {"weakens", "supports"}:
        return SCIENTIFIC_FAILURE
    where = str(row.get("where") or row.get("executor") or "")
    why = str(row.get("why") or row.get("error") or "")
    if where or "parser" in why or "cannot_align" in why or "executor" in where:
        return HARNESS_FAILURE
    if row.get("scientific"):
        return SCIENTIFIC_FAILURE
    return HARNESS_FAILURE if row.get("harness") else SCIENTIFIC_FAILURE


def accumulate_pathology(
    store: dict[str, Pathology],
    *,
    where: str,
    why: str,
    example: str = "",
) -> Pathology:
    key = f"{where}×{why}"
    row = store.get(key)
    if row is None:
        row = Pathology(where=where, why=why, count=0)
        store[key] = row
    row.count += 1
    if example and example not in row.examples:
        row.examples.append(example)
    return row


def pathology_ready(store: Mapping[str, Pathology], *, floor: int = PATHOLOGY_FLOOR) -> list[Pathology]:
    return [row for row in store.values() if row.count >= floor]


def propose_patch(pathology: Pathology) -> CandidatePatch:
    """Component-scoped proposal. Deterministic for known WHERE×WHY pairs."""
    if (
        pathology.where == "question_observation_executor"
        and pathology.why == "cannot_align_versioned_records"
    ):
        artifact = HarnessArtifact(
            name="compare_validator_snapshots",
            module="question_executor",
            scope="action-kind",
            activation="OBSERVE",
            body=(
                "Align validator-version histories with compare_validator_snapshots "
                "before answering an observational question."
            ),
            supported_pathology=pathology.key,
            version="1",
        )
        return CandidatePatch(
            module="question_executor",
            pathology=pathology.key,
            artifact=artifact,
            capability=deepcopy(SNAPSHOT_CAPABILITY),
            files=("harness/overlays/question_executor.json",),
            justification="observation executor cannot align versioned records",
        )
    artifact = HarnessArtifact(
        name=f"note-{pathology.where}",
        module="context" if pathology.where == "notebook_projection" else "skill_registry",
        scope="task",
        activation="REFLECT",
        body=f"Recorded pathology {pathology.key}; no capability invented.",
        supported_pathology=pathology.key,
    )
    return CandidatePatch(
        module=artifact.module,
        pathology=pathology.key,
        artifact=artifact,
        files=(),
        justification="generic scoped note; credit gate still required",
    )


def evaluate_patch(
    patch: CandidatePatch,
    *,
    failing: list[Mapping[str, Any]] | None = None,
    nearby: list[Mapping[str, Any]] | None = None,
    regression: list[Mapping[str, Any]] | None = None,
    previous: list[Mapping[str, Any]] | None = None,
    sealed: list[Mapping[str, Any]] | None = None,
    cost: Mapping[str, Any] | None = None,
) -> CreditReport:
    """Offline fixture comparison for the snapshot helper, not general self-evolution.

    Caller-supplied regression flags and heldout cases are not independently
    attested. Research-time installation still requires the mission credit path.
    """
    reasons: list[str] = []
    compile_ok = True
    try:
        assert_mutable_module(patch.module)
        assert_patch_files(patch.files)
    except KernelViolation as exc:
        compile_ok = False
        reasons.append(str(exc))
    if patch.capability is None and not patch.artifact.body:
        compile_ok = False
        reasons.append("empty patch")

    supported = bool(patch.capability
                     and patch.capability.executor == SNAPSHOT_CAPABILITY.executor
                     and patch.capability.validator == SNAPSHOT_CAPABILITY.validator
                     and patch.capability.version == SNAPSHOT_CAPABILITY.version
                     and patch.module == "question_executor")
    if not supported:
        compile_ok = False
        reasons.append("no evaluator for this patch capability and version")

    def outcomes(cases: list[Mapping[str, Any]] | None) -> tuple[list[bool], list[bool]]:
        before, after = [], []
        for row in cases or []:
            want = row.get("aligned", True)
            valid = type(want) is bool and isinstance(row.get("records"), list)
            # Disabled helper cannot align versions. Negative cases receive
            # exactly one hit, independent of how many versions they contain.
            before.append(valid and not want)
            try:
                observed = run_capability(patch.capability.executor, records=row["records"]) if supported and valid else None
                ok = isinstance(observed, Mapping) and type(observed.get("aligned")) is bool and observed["aligned"] == want
            except (KeyError, TypeError, ValueError, RuntimeError):
                ok = False
            after.append(ok and (not row.get("must_not_break") or row.get("baseline_ok") is True))
        return before, after

    baseline, replay = outcomes(failing)
    failing_ok = bool(replay) and all(replay)
    if not failing_ok:
        reasons.append("failing fixture did not replay")
    nearby_ok = all(outcomes(nearby)[1])
    if nearby and not nearby_ok:
        reasons.append("nearby cases failed")
    regression_ok = True
    for row in regression or []:
        if row.get("must_pass") is False or row.get("passed") is False:
            regression_ok = False
            reasons.append(f"regression: {row.get('name') or 'unnamed'}")
    previous_ok = True
    for row in previous or []:
        if row.get("passed") is False or row.get("must_pass") is False:
            previous_ok = False
            reasons.append(f"previous capability lost: {row.get('name') or 'unnamed'}")
    sealed_gain: bool | None = None
    if sealed:
        sealed_before, sealed_after = outcomes(sealed)
        sealed_gain = all(sealed_after) and sum(sealed_after) > sum(sealed_before)
        if not sealed_gain:
            reasons.append("sealed case shows no generalization")
    cost_row = dict(cost or {})
    try:
        tokens, cap = float(cost_row.get("tokens", 0)), float(cost_row.get("token_cap", 1e9))
        cost_ok = not cost_row.get("bloat") and math.isfinite(tokens) and math.isfinite(cap) and 0 <= tokens <= cap
    except (TypeError, ValueError):
        cost_ok = False
    if not cost_ok:
        reasons.append("cost/bloat check failed")
    h0 = sum(baseline) / len(baseline) if baseline else 0.0
    h1 = sum(replay) / len(replay) if replay else 0.0
    h1_off = h0
    paired_gain = h1 - h0
    artifact_delta = h1 - h1_off
    return CreditReport(
        compile_ok=compile_ok,
        failing_replay=bool(failing_ok),
        nearby_ok=bool(nearby_ok),
        regression_ok=regression_ok,
        previous_ok=previous_ok,
        sealed_gain=sealed_gain,
        cost_ok=cost_ok,
        general=False,  # caller-supplied fixtures do not establish independent generalization
        reasons=tuple(reasons),
        paired_gain=paired_gain,
        artifact_enabled_delta=artifact_delta,
        h0_score=h0,
        h1_score=h1,
        h1_disabled_score=h1_off,
        candidate_digest=patch.digest(),
        evaluation_scope="validator_snapshot_fixtures_v1" if supported else "unsupported",
    )


def apply_admitted_patch(
    harness: HarnessVersion,
    registry: CapabilityRegistry,
    patch: CandidatePatch,
    report: CreditReport,
) -> tuple[HarnessVersion, CapabilityRegistry, str]:
    if not report.admit or report.candidate_digest != patch.digest():
        return harness, registry, "rejected"
    child = admit_artifact(harness, patch.artifact, files=patch.files)
    if patch.capability is not None:
        cap = deepcopy(patch.capability)
        cap.trust_level = "validated"
        cap.evidence_permission = cap.evidence_permission or "evidence-producing"
        cap.provenance = f"harness:{child.version_id}:{patch.pathology}"
        registry.add(cap)
    return child, registry, "admitted"


def load_pathologies(workspace: Path | None) -> dict[str, Pathology]:
    if workspace is None:
        return {}
    path = Path(workspace) / PATHOLOGY_FILE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    store: dict[str, Pathology] = {}
    for item in payload.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        row = Pathology(
            where=str(item.get("where") or ""),
            why=str(item.get("why") or ""),
            count=int(item.get("count") or 0),
            examples=list(item.get("examples") or []),
        )
        store[row.key] = row
    return store


def save_pathologies(workspace: Path | None, store: Mapping[str, Pathology]) -> Path | None:
    if workspace is None:
        return None
    dest = Path(workspace)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / PATHOLOGY_FILE
    path.write_text(
        json.dumps(
            {"items": [row.to_dict() for row in store.values()]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
