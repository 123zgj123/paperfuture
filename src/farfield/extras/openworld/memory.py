"""Working memory, scientific archive, harness archive, failure archive."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .kernel import refuse_generated_promotion


ARCHIVE_DIR = "archives"


def lineage_learning(state: Mapping[str, Any], *, lineage_id: str,
                     parent_id: str = "", limit: int = 6) -> dict[str, Any]:
    """Project ancestor-specific lessons; never write a shared skill or plugin.

    Feedback supplies provisional procedures, not validated improvements. A
    branch can inherit its explicitly named parent's ancestry, not siblings.
    Every packet is rebuilt from this workspace's records; no global catalog
    is consulted and no remembered assertion becomes scientific evidence.
    """
    epoch = int(state.get('project_epoch') or 0)
    from .dependencies import dependency_view
    dependencies = dependency_view(state)
    artifacts = [r for r in state.get('artifacts') or []
                 if int(r.get('project_epoch') or 0) == epoch and r.get('status') != 'retired']
    ideas = {r['id']: r for r in artifacts if r.get('kind') == 'discovery_idea' and r.get('id')}
    feedback = {r.get('idea_id'): r for r in artifacts if r.get('kind') == 'idea_feedback'}
    chain, seen = [], set()
    current = str(parent_id or '')
    for _ in range(max(0, min(limit, 12))):
        if not current or current in seen or current not in ideas:
            break
        seen.add(current)
        idea = ideas[current]
        fb = feedback.get(current) or {}
        measurements = [r for r in state.get('evidence_records', [])
                        if r.get('discovery_idea_id') == current and r.get('epistemic') == 'WORLD'
                        and r.get('attested') and r.get('execution_status') == 'ran'
                        and not r.get('invalidated')]
        chain.append({'idea_id': current, 'lineage_id': idea.get('lineage_id'),
                      'dependency_status': dependencies.get(current, {}),
                      'claim': (idea.get('idea') or {}).get('central_claim', ''),
                      'critic_feedback': {k: fb.get(k) for k in (
                          'id', 'root_failure_cause', 'what_survives', 'what_failed',
                          'mistakes_to_avoid', 'required_evidence', 'recommended_next_operator')},
                      'feedback_basis': 'GENERATED_not_empirical_validation',
                      'measurements': copy.deepcopy(measurements),
                      'exploratory_observations': copy.deepcopy([r for r in artifacts
                          if r.get('kind') == 'exploration_execution' and r.get('idea_id') == current]),
                      'measurement_basis': 'execution_records_retain_validity_checks_not_automatic_support'})
        current = str(idea.get('parent_id') or '')
    chain.reverse()
    skills = [{'source_idea_id': r['idea_id'], 'source_feedback_id': r['critic_feedback'].get('id'),
               'procedure': r['critic_feedback'].get('recommended_next_operator'),
               'avoid': r['critic_feedback'].get('mistakes_to_avoid') or [],
               'status': 'provisional_prompt_advice', 'effectiveness': 'not_established',
               'activation': 'this_lineage_only', 'executable': False}
              for r in chain if r['critic_feedback'].get('recommended_next_operator')]
    body = {'lineage_id': lineage_id, 'project_epoch': epoch, 'parent_id': parent_id,
            'memory': chain, 'skill_candidates': skills}
    return {**copy.deepcopy(body), 'version': 'lineage-learning-v1',
            'digest': hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest(),
            'scope': 'current_workspace_and_explicit_ancestry',
            'shared_installation': False, 'kernel_mutation_allowed': False,
            'transfer_requires': 'separate_heldout_evaluation_and_explicit_version_pin'}


@dataclass
class ArchiveStore:
    working: list[dict[str, Any]] = field(default_factory=list)
    scientific: list[dict[str, Any]] = field(default_factory=list)
    harness: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "working": list(self.working),
            "scientific": list(self.scientific),
            "harness": list(self.harness),
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ArchiveStore":
        row = dict(payload or {})
        return cls(
            working=[dict(item) for item in row.get("working") or [] if isinstance(item, Mapping)],
            scientific=[dict(item) for item in row.get("scientific") or [] if isinstance(item, Mapping)],
            harness=[dict(item) for item in row.get("harness") or [] if isinstance(item, Mapping)],
            failures=[dict(item) for item in row.get("failures") or [] if isinstance(item, Mapping)],
        )


def add_working(store: ArchiveStore, row: Mapping[str, Any]) -> None:
    payload = dict(row)
    payload.setdefault("epistemic", "GENERATED")
    store.working.append(payload)


def add_scientific(store: ArchiveStore, row: Mapping[str, Any]) -> str:
    payload = dict(row)
    blocked = refuse_generated_promotion(payload)
    if blocked and str(payload.get("epistemic") or "") == "WORLD":
        payload["epistemic"] = "GENERATED"
        payload["promotion_refused"] = blocked
        store.working.append(payload)
        return "working"
    if str(payload.get("epistemic") or "") != "WORLD" and not payload.get("attested"):
        store.working.append(payload)
        return "working"
    store.scientific.append(payload)
    return "scientific"


def add_harness(store: ArchiveStore, row: Mapping[str, Any]) -> None:
    store.harness.append(dict(row))


def add_failure(store: ArchiveStore, row: Mapping[str, Any]) -> None:
    store.failures.append(dict(row))


def _world_compatible(row: Mapping[str, Any], current_world: str) -> bool:
    wid = str(row.get("world_id") or "")
    if not wid or not current_world:
        return True
    if wid == current_world:
        return True
    valid = row.get("valid_in_worlds") or []
    if current_world in {str(item) for item in valid}:
        return True
    return False


def _stamp_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.setdefault("world_id", payload.get("world_id") or "")
    payload.setdefault("evidence_id", payload.get("evidence_id") or "")
    payload.setdefault("epistemic", payload.get("epistemic") or "GENERATED")
    return payload


def project_archive(
    store: ArchiveStore,
    *,
    action_type: str,
    limit: int = 6,
    world_id: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve the slice this action needs. Never the full history."""
    action = str(action_type or "").upper()
    working = list(store.working[-limit:])
    scientific = [
        _stamp_projection(row)
        for row in store.scientific
        if _world_compatible(row, world_id)
    ][-limit:]
    stale = [
        _stamp_projection(row)
        for row in store.scientific
        if world_id and not _world_compatible(row, world_id)
    ][-limit:]
    harness = list(store.harness[-limit:])
    failures = list(store.failures[-limit:])
    if action in {"SURVEY", "SYNTHESIZE", "ARCHIVE"}:
        harness = []
        failures = failures[-2:]
    if action == "PROBE":
        working = [row for row in working if row.get("kind") in {None, "probe", "question"}][:limit]
        harness = [row for row in harness if "probe" in str(row.get("module") or "").lower()]
    if action == "OBSERVE":
        harness = [row for row in harness if "question" in str(row.get("module") or row.get("name") or "").lower()]
        working = [row for row in working if row.get("kind") in {None, "question", "observe"}][:limit]
    if action == "EVOLVE_HARNESS":
        scientific = []
        working = []
    if action == "ACQUIRE":
        harness = [row for row in harness if "acquire" in str(row.get("module") or "").lower()]
    return {
        "working": [_stamp_projection(row) for row in working],
        "scientific": scientific,
        "harness": harness,
        "failures": failures,
        "stale": stale,
    }


def projection_token_count(slice_: Mapping[str, Any]) -> int:
    raw = json.dumps(dict(slice_), ensure_ascii=False)
    return len(raw.split())


def load_archives(workspace: Path | None) -> ArchiveStore:
    if workspace is None:
        return ArchiveStore()
    path = Path(workspace) / ARCHIVE_DIR / "index.json"
    if not path.is_file():
        return ArchiveStore()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ArchiveStore()
    if not isinstance(payload, dict):
        return ArchiveStore()
    return ArchiveStore.from_dict(payload)


def save_archives(workspace: Path | None, store: ArchiveStore) -> Path | None:
    if workspace is None:
        return None
    dest = Path(workspace) / ARCHIVE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "index.json"
    path.write_text(
        json.dumps(store.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
