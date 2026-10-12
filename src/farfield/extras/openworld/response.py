"""Require an explicit response to parent-local results before further ideation.

This checks reference coverage and freshness, not the truth of a rationale.
Receipts stay GENERATED; they cannot promote evidence or update a belief.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .inquiry import get, inquiry_request, text


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def result_packet(state: Any, parent_id: str) -> dict[str, Any]:
    """Project canonical results; never accept a worker-supplied observation list."""
    epoch = int(get(state, 'project_epoch', 0) or 0)
    artifacts = [r for r in get(state, 'artifacts', [])
                 if int(r.get('project_epoch') or 0) == epoch]
    parent = next((r for r in artifacts if r.get('kind') == 'discovery_idea'
                   and r.get('id') == parent_id), None)
    results = []
    from .dependencies import dependency_view
    dependencies = dependency_view(state)
    if parent:
        from .exploratory import records
        for r in records(state, parent_id):
            results.append({'result_id': 'exploration:' + r['id'], 'digest': digest(r),
                'kind': 'unvalidated_exploration', 'observation': dict(r),
                'dependency_status': dependencies.get(r['id'], {}),
                'interpretation_constraints': [
                    'successful_execution_does_not_validate_the_instrument',
                    'readable_files_and_row_counts_do_not_prove_data_integrity',
                    'parsing_schema_units_and_controls_require_checks',
                    'performance_and_mechanism_remain_separate_unassessed_questions']})
        for r in artifacts:
            if r.get('kind') in ('native_evaluation', 'native_comparison') and r.get('idea_id') == parent_id:
                results.append({'result_id': 'evaluation:' + r['id'], 'digest': digest(r),
                    'kind': 'independent_native_evaluation', 'observation': dict(r),
                    'dependency_status': dependencies.get(r['id'], {}),
                    'interpretation_constraints': ['score_applies_only_to_frozen_metric_contract',
                                                   'aggregate_score_does_not_localize_the_error',
                                                   'use_metric_feedback_to_separate_pipeline_components',
                                                   'unchanged_target_is_not_an_effective_mechanism_intervention',
                                                   'method_performance_does_not_identify_mechanism']})
        affected = set(dependencies.get(parent_id, {}).get('invalidated_by', []))
        for result in results:
            affected.update(result.get('dependency_status', {}).get('invalidated_by', []))
        latest = {r['target_id']: r for r in artifacts
                  if r.get('kind') == 'research_retraction' and r.get('target_id') in affected}
        for r in latest.values():
            results.append({'result_id': 'retraction:' + r['target_id'], 'digest': digest(r),
                'kind': 'dependency_retraction', 'observation': dict(r)})
        for r in get(state, 'evidence_records', []):
            if (r.get('discovery_idea_id') == parent_id or r.get('theory_id') == parent_id) and r.get('evidence_id'):
                results.append({'result_id': 'evidence:' + r['evidence_id'],
                    'digest': digest(r), 'kind': 'experiment_record',
                    'observation': {k: r[k] for k in ('evidence_id', 'outcome', 'epistemic',
                        'attested', 'execution_status', 'verification', 'invalidated',
                        'validity', 'measurement', 'metrics', 'result', 'effect', 'summary',
                        'registration', 'invalidation_reason', 'failure_type', 'promotion_refused',
                        'cannot_corroborate', 'study_stage', 'verdict', 'treatment', 'control',
                        'sample_size', 'sample_size_basis', 'primary_metric', 'inference_link',
                        'failure_diagnosis', 'reproduction_ok', 'stage_verdict', 'pilot_verdict') if k in r}})
        surveys = [r for r in artifacts if r.get('kind') == 'literature_survey'
                   and (r.get('knowledge_request') or {}).get('idea_id') == parent_id]
        for r in surveys:
            results.append({'result_id': 'survey:' + str(r.get('id') or digest(r)),
                'digest': digest(r), 'kind': 'literature_result',
                'observation': {k: r[k] for k in ('knowledge_request', 'citations',
                    'source_check', 'source_check_gap', 'technical_map', 'technical_map_gap',
                    'document_gaps') if k in r}})
        feedback = [r for r in artifacts if r.get('kind') == 'idea_feedback' and r.get('idea_id') == parent_id]
        requests = {inquiry_request(parent, f).get('id') for f in feedback} - {None}
        requests.update({inquiry_request(parent, {}).get('id')} - {None})
        completed = {(r.get('knowledge_request') or {}).get('id') for r in surveys}
        for r in artifacts:
            if (r.get('kind') == 'action_attempt' and r.get('action_type') == 'EXPLORE'
                    and r.get('target') == parent_id and not records(state, parent_id)):
                results.append({'result_id': 'attempt:' + digest(r), 'digest': digest(r),
                    'kind': 'exploration_attempt_without_observation',
                    'observation': {k: r.get(k) for k in ('status', 'reason', 'target')}})
            if (r.get('kind') == 'action_attempt' and r.get('action_type') == 'PROBE'
                    and parent_id in {r.get('discovery_idea_id'), r.get('theory_id')}
                    and not r.get('evidence_id')):
                results.append({'result_id': 'attempt:' + digest(r), 'digest': digest(r),
                    # Older logs omit evidence_id even after successful runs.
                    # Missing linkage must not assert missing observations.
                    'kind': 'experiment_attempt_without_evidence_link',
                    'observation': {k: r[k] for k in ('target', 'status', 'reason', 'outcome_key',
                        'service_unavailable') if k in r}})
            if (r.get('kind') == 'action_attempt' and r.get('action_type') == 'SURVEY'
                    and r.get('target') in requests - completed):
                results.append({'result_id': 'attempt:' + digest(r), 'digest': digest(r),
                    'kind': 'inquiry_without_sources', 'observation': {k: r[k] for k in
                        ('target', 'status', 'reason', 'outcome_key', 'service_unavailable') if k in r}})
    # Do not silently truncate unaddressed results. Large lineages must pay the
    # reading cost or pause; dropping negative results would bias decisions.
    results = sorted({(r['result_id'], r['digest']): r for r in results}.values(),
                     key=lambda r: (r['result_id'], r['digest']))
    return {'parent_id': parent_id, 'project_epoch': epoch, 'results': results,
            'packet_digest': digest(results), 'required': bool(results)}


def validate_response(packet: Mapping[str, Any], response: Any) -> str:
    if response is None and not packet['required']:
        return ''
    if not isinstance(response, Mapping):
        return 'result_response_required'
    if response.get('packet_digest') != packet['packet_digest']:
        return 'result_response_stale_or_unbound'
    if not isinstance(response.get('disposition'), str) or response['disposition'] not in {'continue', 'revise', 'verify', 'pause'}:
        return 'result_response_invalid_disposition'
    if not text(response.get('rationale')) or not text(response.get('next_step')):
        return 'result_response_reason_and_next_step_required'
    addressed = response.get('addressed_results')
    if not isinstance(addressed, list):
        return 'result_response_coverage_required'
    refs = []
    for row in addressed:
        if (not isinstance(row, Mapping) or not isinstance(row.get('result_id'), str)
                or not isinstance(row.get('digest'), str) or not text(row.get('interpretation'))):
            return 'result_response_interpretation_required'
        refs.append((row['result_id'], row['digest']))
    expected = {(r['result_id'], r['digest']) for r in packet['results']}
    if len(refs) != len(set(refs)) or set(refs) != expected:
        return 'result_response_missing_or_unknown_results'
    by_id = {r['result_id']: r for r in addressed}
    for result in packet['results']:
        if result.get('kind') == 'dependency_retraction':
            action = by_id[result['result_id']].get('dependency_action')
            if action not in ('discard', 'recheck', 'independent_basis', 'defer'):
                return 'result_response_retraction_action_required'
            if action == 'defer' and response['disposition'] != 'pause':
                return 'unresolved_retraction_requires_pause_or_active_review'
        audit = result.get('observation', {}).get('measurement_audit')
        if not audit:
            continue  # Old logs do not retrospectively acquire a new contract.
        assessment = by_id[result['result_id']].get('measurement_assessment')
        if not isinstance(assessment, Mapping):
            return 'result_response_measurement_assessment_required'
        if assessment.get('status') != audit['status']:
            return 'result_response_measurement_status_mismatch'
        # Parser agreement cannot confirm a performance benefit or explanation.
        for key in ('performance', 'mechanism'):
            if assessment.get(key) not in ('not_assessed', 'provisional'):
                return 'result_response_scientific_support_exceeds_measurement_scope'
        failed = {r['check_id'] for r in audit['checks'] if r['status'] == 'failed'}
        replies = assessment.get('failed_checks', [])
        if not isinstance(replies, list):
            return 'result_response_failed_measurements_unaddressed'
        ids = []
        for row in replies:
            if (not isinstance(row, Mapping) or not isinstance(row.get('check_id'), str)
                    or row.get('action') not in ('repair', 'dispute', 'defer')
                    or not text(row.get('reason'))):
                return 'result_response_failed_measurements_unaddressed'
            ids.append(row['check_id'])
        if len(ids) != len(set(ids)) or set(ids) != failed:
            return 'result_response_failed_measurements_unaddressed'
    return ''


def action_pause_reason(state: Any, action: Any) -> str:
    """Pause only named idea actions, until the local result packet changes.

    Verification/reporting remain possible. A pause does not refute the idea,
    stop the project, or erase historical observations.
    """
    if action.action_type in {'VERIFY', 'SYNTHESIZE', 'REVIEW', 'ARCHIVE'}:
        return ''
    ids = {str(action.target)} | {str(action.extra.get(k) or '') for k in
        ('parent_idea_id', 'discovery_idea_id', 'idea_id', 'theory_id')}
    epoch = int(get(state, 'project_epoch', 0) or 0)
    for row in reversed(get(state, 'artifacts', [])):
        if (row.get('kind') == 'discovery_decision' and row.get('parent_id') in ids
                and int(row.get('project_epoch') or 0) == epoch):
            response = row.get('result_response') or {}
            packet = result_packet(state, row['parent_id'])
            if response.get('disposition') == 'pause' and not validate_response(packet, response):
                return 'idea_paused_pending_new_result'
    return ''


RESPONSE_HELP = (
    ' When current_research_state.result_packet.required is true, return result_response: '
    '{packet_digest, disposition (continue|revise|verify|pause), rationale, next_step, '
    'addressed_results:[{result_id,digest,interpretation}]}. Copy the packet digest and '
    'respond individually to EVERY result with its exact ID and digest. Explain how it '
    'affects the next decision; do not merely list citations. An uninformative result or '
    'independent evidence may justify continue without changing the hypothesis. Distinguish '
    'invalid measurement, implementation/environment failure, insufficient power and '
    'hypothesis refutation. Source interpretations are not experimental observations. '
    'For a result with observation.measurement_audit, its addressed_results entry must '
    'include measurement_assessment:{status: copy audit.status, performance: '
    'not_assessed|provisional, mechanism: not_assessed|provisional, '
    'failed_checks:[{check_id,action:repair|dispute|defer,reason}]}. Address every failed '
    'check. A discrepancy concerns the declared measurement, not hypothesis refutation. '
    'Profiles include headers and empty fields: justify exclusions and units separately. '
    'Passing scoped checks does not establish integrity, a performance gain or mechanism. '
    'For each dependency_retraction include dependency_action: discard|recheck|'
    'independent_basis|defer in its addressed_results entry, and explain the consequence. '
    'Defer requires pause. Reviewing a retraction does not restore the old measurement; '
    'a new independent observation is required. Native evaluator scores refer only to '
    'their declared population and metric; do not infer a causal mechanism from a score. '
    'For a provisional mechanism identify a competing explanation and a discriminating '
    'intervention in the next idea; keep a performance comparison separate. '
    'For verify name the missing discriminating check; for pause name what would permit '
    'resumption. Pause returns only result_response, with no new idea; the named idea is '
    'deferred until new local results arrive. Never manufacture a reason to flip a belief.'
)
