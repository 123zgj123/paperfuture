"""Derived dependency status for exploratory executions and native evaluations.

Retractions append history. Raw observations are never erased. A downstream
result is reconsidered, not automatically declared scientifically false.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...services.experiments import file_digest, fingerprint
from .inquiry import get
from .events import make_event

KINDS = {'exploration_execution', 'native_evaluation', 'native_comparison', 'discovery_idea'}


def _resolve(reference: str, nodes: dict) -> str:
    if reference in nodes:
        return reference
    prefixes = {'exploration': ('exploration_execution',),
                'evaluation': ('native_evaluation', 'native_comparison')}
    prefix, separator, ident = reference.partition(':')
    if separator and nodes.get(ident, {}).get('kind') in prefixes.get(prefix, ()):
        return ident
    return reference


def resolve_reference(state: Any, reference: str) -> str:
    nodes = {r['id']: r for r in get(state, 'artifacts', []) if r.get('kind') in KINDS and r.get('id')
             and r.get('project_epoch', 0) == get(state, 'project_epoch', 0)}
    return _resolve(reference, nodes)


def dependency_view(state: Any) -> dict[str, dict]:
    rows = [r for r in get(state, 'artifacts', [])
            if r.get('project_epoch', 0) == get(state, 'project_epoch', 0)]
    nodes = {r['id']: r for r in rows if r.get('kind') in KINDS and r.get('id')}
    edges = {}
    for ident, row in nodes.items():
        refs = list(row.get('depends_on') or [])
        if row['kind'] == 'exploration_execution':
            refs += list((row.get('proposal') or {}).get('depends_on') or [])
            refs += [row.get('idea_id')]
        if row['kind'] == 'discovery_idea':
            refs += [row.get('parent_id')]
            refs += [str(r.get('result_id', '')).split(':', 1)[-1]
                     for r in (row.get('result_response') or {}).get('addressed_results', [])]
        edges[ident] = sorted({_resolve(r, nodes) for r in refs
                               if isinstance(r, str) and _resolve(r, nodes) in nodes})
    revoked = {r['target_id']: r for r in rows if r.get('kind') == 'research_retraction'}
    status = {key: {'status': 'retracted' if key in revoked else 'current',
                   'depends_on': refs, 'invalidated_by': [key] if key in revoked else []}
              for key, refs in edges.items()}
    # Monotone closure also terminates for malformed old graphs containing cycles.
    for _ in range(len(nodes)):
        changed = False
        for key, refs in edges.items():
            causes = sorted(set(status[key]['invalidated_by']).union(
                *(set(status[r]['invalidated_by']) for r in refs)))
            if nodes[key]['kind'] == 'discovery_idea' and key not in revoked:
                reviewed = {str(r.get('result_id', '')).removeprefix('retraction:')
                    for r in (nodes[key].get('result_response') or {}).get('addressed_results', [])
                    if r.get('dependency_action') in ('discard', 'recheck', 'independent_basis')}
                causes = [c for c in causes if c not in reviewed]
            if causes != status[key]['invalidated_by']:
                status[key]['invalidated_by'] = causes
                if key not in revoked:
                    status[key]['status'] = 'reconsideration_required'
                changed = True
        if not changed:
            break
    return status


def reference_error(state: Any, proposal: dict) -> str:
    refs = proposal.get('depends_on', [])
    if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
        return 'result_dependencies_must_be_result_ids'
    statuses = dependency_view(state)
    nodes = {r['id']: r for r in get(state, 'artifacts', []) if r.get('kind') in KINDS and r.get('id')}
    refs = [_resolve(r, nodes) for r in refs]
    if any(r not in statuses for r in refs):
        return 'unknown_result_dependency'
    if any(statuses[r]['status'] != 'current' for r in refs):
        return 'retracted_result_cannot_be_reused_as_valid_basis'
    return ''


def retraction_event(state: Any, target: str, reason: str, *, origin: str):
    target = resolve_reference(state, target)
    if target not in dependency_view(state) or not reason.strip():
        raise ValueError('retraction_requires_known_result_and_reason')
    body = {'kind': 'research_retraction', 'target_id': target, 'reason': reason,
            'origin': origin, 'project_epoch': get(state, 'project_epoch', 0),
            'epistemic': 'GENERATED', 'scientific_falsity': 'not_implied'}
    body['id'] = 'retraction-' + fingerprint(body)[:24]
    return make_event('ArtifactAdded', body)


def integrity_retractions(workspace: Path, state: Any):
    """Recheck saved evaluation inputs on resume; never rescore or overwrite."""
    result = []
    statuses = dependency_view(state)
    for row in get(state, 'artifacts', []):
        if row.get('kind') != 'native_evaluation' or row.get('status') != 'evaluated':
            continue
        if statuses.get(row['id'], {}).get('status') != 'current':
            continue
        root = Path(workspace) / 'native_evaluations' / row['id'] / 'frozen_submission'
        try:
            unchanged = all(not (root / name).is_symlink() and
                file_digest(root / name) == expected for name, expected in row['regenerated_files'].items())
            config = Path(workspace) / 'native_evaluators' / row['contract_digest'] / 'config.json'
            import json
            frozen = json.loads(config.read_text())
            unchanged = unchanged and fingerprint(frozen) == row['contract_digest']
            unchanged = unchanged and all(not (config.parent / name).is_symlink() and
                file_digest(config.parent / name) == expected for name, expected in frozen['assets'].items())
            if row.get('observation_file_digest'):
                observation = root.parent / 'evaluator' / frozen['public']['observation_file']
                unchanged = unchanged and not observation.is_symlink() and (
                    file_digest(observation) == row['observation_file_digest'])
        except (OSError, ValueError, KeyError):
            unchanged = False
        if not unchanged:
            result.append(retraction_event(state, row['id'], 'saved_evaluation_inputs_or_contract_changed',
                                           origin='host_integrity_recheck'))
    return result


def retract_result(workspace: Path, target: str, reason: str) -> dict:
    """Explicit host/operator API; never exposed as a solver-controlled tool."""
    from ...services.workspace import exclusive_workspace
    from . import OpenWorld
    from .reducer import apply_events
    with exclusive_workspace(workspace):
        env = OpenWorld.load(workspace)
        event = retraction_event(env.state, target, reason, origin='host_operator')
        env.state = apply_events(env.state, [event], log=env.log)
        env.persist()
        return dependency_view(env.state)
