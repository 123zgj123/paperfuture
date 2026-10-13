"""Run ownership, idea lineage and explicit transfer of conditional lessons."""
from __future__ import annotations

import copy

from .contracts import ContractError


def scoped_feedback(feedback, scope):
    """Use the same branch boundary in proposal and source-tool planning."""
    visible = copy.deepcopy(feedback)
    if scope and visible and visible['action'] == 'switch':
        visible.pop('parent_idea', None)
        visible.pop('parent_unknowns', None)
    return visible


def round_scope(run_id, number, feedback, previous):
    by_number = {r['number']:r for r in previous}
    for row in previous:
        scope = row.get('scope', {})
        if scope.get('run_id') != run_id or scope.get('idea_id') != f"{run_id}/idea/{row['number']:02}":
            raise ContractError('foreign or unowned idea artifact in run memory')
    action = (feedback or {}).get('action', 'initial')
    parent = by_number.get((feedback or {}).get('parent_round'))
    if feedback and parent is None:
        raise ContractError('memory parent must belong to this run')
    inherits = parent is not None and action in ('revise', 'continue_leap')
    ancestors = []
    if inherits:
        ancestors = [*parent['scope']['ancestor_idea_ids'], parent['scope']['idea_id']]
    return {'run_id':run_id, 'idea_id':f'{run_id}/idea/{number:02}',
        'branch_id':parent['scope']['branch_id'] if inherits else f'{run_id}/branch/{number:02}',
        'parent_idea_id':parent['scope']['idea_id'] if inherits else None,
        'inspiration_idea_id':parent['scope']['idea_id'] if parent and not inherits else None,
        'ancestor_idea_ids':ancestors, 'action':action,
        'policy':'ancestor conditions require explicit retain; other branches require an applicability record; no cross-run memory'}


def memory_view(experiences, scope, feedback):
    """Full lessons from ancestry or explicit navigation; other branches indexed."""
    if not scope:
        return experiences, []  # Legacy direct-function callers, not new runs.
    requested = set((feedback or {}).get('navigation',{}).get('experience_ids',[]))
    available = {e['id'] for e in experiences}
    if requested-available:
        raise ContractError('navigation refers to unavailable experience memory')
    active = []; index = []
    ancestors = set(scope['ancestor_idea_ids'])
    for experience in experiences:
        if experience.get('source_run_id') != scope['run_id']:
            raise ContractError('cross-run experience memory is forbidden')
        if experience['source_idea_id'] in ancestors or experience['id'] in requested:
            active.append(experience)
        else:
            index.append({k:experience[k] for k in ('id','round','kind','source_idea_id','source_branch_id','epistemic_status')})
    return active, index


def validate_transfers(proposal, experiences, scope, evidence):
    if not scope:
        return
    by_id = {e['id']:e for e in experiences}
    used = {row['experience_id'] for row in proposal['learning_trace'] if row['usage']!='not_applicable'}
    ancestors=set(scope['ancestor_idea_ids'])
    ancestry_conditions={a['id'] for e in experiences if e['source_idea_id'] in ancestors for a in e['assumptions']}
    active={a['id'] for a in proposal['assumptions']} | {
        row['assumption_id'] for row in proposal['assumption_decisions'] if row['disposition']=='retain'}
    outside={e['id']:{a['id'] for a in e['assumptions']} for e in experiences if e['source_idea_id'] not in ancestors}
    foreign_conditions=(set().union(*outside.values()) if outside else set())-ancestry_conditions
    accounted=set().union(*(outside[key] for key in used if key in outside)) if used else set()
    if active & foreign_conditions - accounted:
        raise ContractError('retained cross-branch conditions require a cited experience and explicit memory transfer')
    required = {key for key in used if by_id[key]['source_idea_id'] not in ancestors}
    rows = proposal.get('memory_transfers',[])
    if len(rows) != len({r['experience_id'] for r in rows}) or {r['experience_id'] for r in rows} != required:
        raise ContractError('each used lesson outside idea ancestry requires exactly one explicit memory transfer')
    from .contracts import require_refs
    for row in rows:
        require_refs(row['evidence_ids'], evidence, 'memory applicability evidence')
        if not row['applicability'].strip() or not row['test'].strip():
            raise ContractError('memory transfer needs applicability conditions and a discriminating test')
