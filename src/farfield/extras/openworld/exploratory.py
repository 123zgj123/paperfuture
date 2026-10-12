"""Task-local, open Python exploration through the existing isolated lab.

Execution and printed measurements are observations, not validated scientific
evidence. No world admission, belief promotion or skill publication occurs here.
The host selects visible assets; the researcher selects questions and programs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from ...services.experiments import canonical, file_digest, fingerprint
from .events import make_event
from .inquiry import get
from .kernel import CandidateResult

EXPLORE = 'EXPLORE'
RESULT_KIND = 'exploration_execution'
ENVIRONMENT_KIND = 'exploration_environment'

EXPLORATION_HELP = (
    ' When current_research_state.exploration_environment is present, you may return '
    'exploration:{question, purpose, code, args:{}, prediction:{}}. This runs arbitrary '
    'Python in the task-local isolated lab, not a menu of algorithms. Code reads the JSON '
    'file sys.argv[1]; initial assets are read-only files under os.environ["LAB_DATA"]. '
    'Use installed scientific packages and print RESULT followed by a JSON object of '
    'finite numeric measurements. Logs may contain qualitative inspection. Network is '
    'off; only the run folder is writable. Inspect initial assets before committing to '
    'a specific research question. Early hypotheses may be provisional and predictions '
    'empty: do not invent intervals or positive-effect thresholds to gain execution. '
    'For exploratory work you may return only exploration plus the required '
    'result_response; you need not write a novelty, impact or mechanism narrative '
    'before inspecting or testing. Such a proposal remains a provisional question. '
    'purpose may describe inspection, instrument building, method development, a '
    'performance comparison or a mechanism-discriminating intervention. You write the '
    'implementation; never silently substitute a prebuilt method for a proposed mechanism. '
    'Each run freezes its code, inputs and forecast, not the entire research agenda. '
    'Printed scores are unvalidated observations, not confirmation or mechanism proof. '
    'For table counts you can add measurement_bindings:[{metric,asset,quantity:record_count, '
    'header_records:0}] or {metric,asset,quantity:field_count,record_number:1}. Nonzero '
    'header_records requires header_basis. A host-owned CSV/TSV parser checks these '
    'definitions against the assets and returns structure profiles even without bindings. '
    'Header meaning and scientific units remain assumptions; do not invent a binding '
    'for unsupported measurements. Inspection is allowed without bindings. '
    'Separate performance tests from mechanism tests; preserve alternative explanations. '
    'Use actual failures to revise the next program. An unchanged program and arguments '
    'are not a fresh independent replication. To deliberately repeat, add repeat_of with '
    'the prior exploration ID and repeat_reason (e.g. reproducibility or environment repair). '
    'Repeats cost budget and do not add independent units. Leave experiment empty for exploration; '
    'formal confirmation still needs a separate frozen protocol and independent evaluator.'
    ' If current_research_state.native_evaluator is present, implement its public file '
    'interface in your working directory and set exploration.evaluate=true to request '
    'clean replay and independent scoring. Its private data and evaluator code are not '
    'available to your program. Inspection need not request evaluation. Declare '
    'exploration.depends_on:[result IDs] for prior results used as a valid scientific '
    'basis. Retracted results cannot be reused as valid premises; they may motivate new tests.'
    ' Set exploration.compare_to to a prior native evaluation ID for a score contrast '
    'under the identical frozen metric contract. A contrast evaluates performance on '
    'those cases, not the proposed causal explanation or generalization.'
    ' To test a mechanism component, preregister exploration.intervention:{'
    'target_observables:[names], invariant_observables:[names], prediction:text} together '
    'with compare_to. Use only host-registered evaluator observable names. The evaluator '
    'checks whether the target actually changed and declared controls stayed fixed; '
    'changing code or obtaining a higher score alone does not establish this.'
)


def environment(state: Any) -> dict[str, Any]:
    rows = [r for r in get(state, 'artifacts', []) if r.get('kind') == ENVIRONMENT_KIND]
    return dict(rows[-1]) if rows else {}


def records(state: Any, idea_id: str) -> list[dict[str, Any]]:
    return [r for r in get(state, 'artifacts', []) if r.get('kind') == RESULT_KIND
            and r.get('idea_id') == idea_id
            and r.get('project_epoch', 0) == get(state, 'project_epoch', 0)]


def attempts(state: Any, idea_id: str) -> list[dict[str, Any]]:
    return [r for r in get(state, 'artifacts', []) if r.get('kind') == 'action_attempt'
            and r.get('action_type') == EXPLORE and r.get('target') == idea_id
            and r.get('project_epoch', 0) == get(state, 'project_epoch', 0)]


def bind_assets(workspace: Path, assets: Mapping[str, Path]) -> dict[str, Any]:
    """Snapshot explicitly exposed files, refusing silent changes on resume."""
    manifest = {}
    for name, raw in assets.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('invalid_exploration_asset_name')
        source = Path(raw)
        if source.is_symlink() or not source.is_file():
            raise ValueError('exploration_asset_must_be_a_regular_file')
        manifest[name] = {'sha256': file_digest(source), 'bytes': source.stat().st_size}
    identity = fingerprint(manifest)
    folder = Path(workspace) / 'exploration_assets' / identity
    folder.mkdir(parents=True, exist_ok=True)
    for name, raw in assets.items():
        dest = folder / name
        if not dest.exists():
            with dest.open('xb') as output, Path(raw).open('rb') as source:
                shutil.copyfileobj(source, output)
        if dest.is_symlink() or file_digest(dest) != manifest[name]['sha256']:
            raise ValueError('exploration_asset_snapshot_changed')
    return {'kind': ENVIRONMENT_KIND, 'id': 'assets-' + identity[:20],
            'assets_digest': identity, 'assets': manifest,
            'execution': 'isolated_python', 'network': False, 'gpu': False,
            'timeout_seconds': 60, 'measurement_validation': 'bounded_table_structure_and_explicit_bindings',
            'learning_scope': 'this_workspace_only', 'epistemic': 'GENERATED'}


def proposal_error(proposal: Any) -> str:
    if not isinstance(proposal, Mapping):
        return 'exploration_proposal_must_be_an_object'
    for key in ('question', 'purpose', 'code'):
        if not isinstance(proposal.get(key), str) or not proposal[key].strip():
            return 'exploration_requires_' + key
    if not isinstance(proposal.get('args', {}), Mapping) or not isinstance(proposal.get('prediction', {}), Mapping):
        return 'exploration_args_and_prediction_must_be_objects'
    try:
        if len(canonical(dict(proposal))) > 512_000:
            return 'exploration_payload_too_large'
    except (TypeError, ValueError):
        return 'exploration_payload_not_finite_json'
    return ''


def program_logs(folder: Path) -> dict[str, Any]:
    """Retain beginning and end of host-captured logs, with explicit omissions.

    A transcript is generated content, never an independent measurement check.
    Full logs remain outside the program's writable directory for local review.
    """
    logs = {}
    for name in ('stdout', 'stderr'):
        path = folder / ('work.' + name)
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            size = path.stat().st_size
            head = stream.read(4096)
            stream.seek(max(len(head), size - 2500))
            tail = stream.read(2500)
        logs[name] = {'head': head.decode('utf-8', errors='replace'),
                      'tail': tail.decode('utf-8', errors='replace'), 'bytes': size,
                      'omitted_bytes': max(0, size - len(head) - len(tail)),
                      'sha256': file_digest(path), 'origin': 'untrusted_program_output'}
    return logs


class ExplorationWorkers:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)

    def executors(self):
        return {EXPLORE: self.explore}

    def explore(self, action: Any, state: Any, **kwargs: Any) -> CandidateResult:
        from ...scientist import lab

        env = environment(state)
        idea = next((r for r in get(state, 'artifacts', [])
                     if r.get('kind') == 'discovery_idea' and r.get('id') == action.target), {})
        proposal = (idea.get('idea') or {}).get('exploration')
        error = proposal_error(proposal)
        from .dependencies import reference_error, dependency_view
        if not error:
            error = reference_error(state, proposal)
        if dependency_view(state).get(action.target, {}).get('status') not in (None, 'current'):
            error = 'idea_requires_response_to_retracted_basis'
        if not env or error or records(state, action.target):
            return CandidateResult(status='blocked', extras={'reason': error or
                'exploration_unavailable_or_already_attempted', 'measured_cost': {'api_calls': 0}})
        root = self.workspace / 'exploration_assets' / env['assets_digest']
        for name, expected in env['assets'].items():
            path = root / name
            if path.is_symlink() or not path.is_file() or file_digest(path) != expected['sha256']:
                return CandidateResult(status='blocked', extras={'reason': 'exploration_assets_changed',
                                                                 'measured_cost': {'api_calls': 0}})
        spec = json.loads(canonical(dict(proposal)))
        execution_key = fingerprint({'code': spec['code'], 'args': spec.get('args', {}),
                                     'assets_digest': env['assets_digest']})
        recorded = {r.get('id') for r in get(state, 'artifacts', []) if r.get('kind') == RESULT_KIND}
        for path in (self.workspace / 'explorations').glob('*/proposal.json'):
            frozen = json.loads(path.read_text())
            if (frozen.get('idea_id') == action.target and frozen.get('id') not in recorded
                    and frozen.get('project_epoch') == get(state, 'project_epoch', 0)):
                # A prior process may have run before its event was committed.
                # Import uncertainty, never silently rerun it after a restart.
                pending = {**frozen, 'status': 'execution_unknown',
                           'reason': 'interrupted_attempt_requires_receipt_inspection',
                           'scientific_verdict': 'not_assessed', 'measurement_validity': 'unverified'}
                return CandidateResult(status='execution_unknown', events=[make_event(
                    'ArtifactAdded', pending, action_type=EXPLORE)],
                    extras={'measured_cost': {'api_calls': 0}})
        previous = next((r for r in get(state, 'artifacts', [])
                         if r.get('kind') == RESULT_KIND and r.get('execution_key') == execution_key), None)
        rid = 'explore-' + uuid.uuid4().hex
        row = {'id': rid, 'kind': RESULT_KIND, 'idea_id': action.target,
               'lineage_id': idea.get('lineage_id', ''), 'project_epoch': get(state, 'project_epoch', 0),
               'execution_key': execution_key, 'proposal': spec, 'assets_digest': env['assets_digest'],
               'epistemic': 'GENERATED', 'scientific_verdict': 'not_assessed',
               'measurement_validity': 'unverified', 'mechanism_verdict': 'not_assessed'}
        folder = self.workspace / 'explorations' / rid
        folder.mkdir(parents=True, exist_ok=False)
        # Freeze forecasts as well as code BEFORE execution. This file lives
        # outside the child-writable work directory, even if the run is interrupted.
        registration = {**row, 'registered_at': datetime.now(timezone.utc).isoformat()}
        with (folder / 'proposal.json').open('xb') as handle:
            handle.write(canonical(registration) + b'\n')
        row['proposal_digest'] = fingerprint(registration)
        intentional_repeat = bool(previous and spec.get('repeat_of') == previous['id']
            and isinstance(spec.get('repeat_reason'), str) and spec['repeat_reason'].strip())
        if intentional_repeat:
            row.update(repeat_of=previous['id'], independent_replication=False)
        if previous and not intentional_repeat:
            row.update(status='duplicate_not_executed', previous_result_id=previous['id'],
                       reason='unchanged_code_arguments_and_assets_are_not_new_information')
        else:
            try:
                result = lab.run_once(spec['code'], spec.get('args', {}),
                    workdir=folder / 'work',
                    timeout=env['timeout_seconds'], data_root=root, input_files=sorted(env['assets']))
                row.update(status='observed' if result.ok else 'execution_failed', run=result.to_dict(),
                           observation_origin='isolated_program_output',
                           failure_class=('environment_failure' if result.failure in
                             {'sandbox_unavailable', 'no_stage_for_the_sandbox'} else
                             'execution_or_measurement_failure' if not result.ok else 'unassessed'))
                row['program_logs'] = program_logs(folder)
                if result.ok:
                    from .measurement import audit_measurements
                    try:
                        audit = audit_measurements(root, env['assets'], spec, result.result)
                    except (OSError, ValueError) as exc:
                        audit = {'status': 'unverified', 'checks': [],
                                 'error_type': type(exc).__name__,
                                 'reason': 'independent_measurement_checker_unavailable'}
                    row['measurement_audit'] = audit
                    row['measurement_validity'] = audit['status']
                    if audit['status'] == 'failed':
                        row['failure_class'] = 'measurement_invalid'
            except (OSError, ValueError) as exc:
                row.update(status='execution_failed', failure_class='environment_or_input_failure',
                           error_type=type(exc).__name__)
        events = [make_event('ArtifactAdded', row, action_type=EXPLORE)]
        if row['status'] == 'observed' and spec.get('evaluate') is True:
            evaluators = [r for r in get(state, 'artifacts', []) if r.get('kind') == 'native_evaluation_environment']
            if evaluators:
                from ...services.native_evaluation import evaluate_program
                try:
                    evaluated = evaluate_program(self.workspace, evaluators[-1], row, root,
                                                 sorted(env['assets']), folder)
                except (OSError, ValueError) as exc:
                    evaluated = {'kind': 'native_evaluation', 'id': 'evaluation-' + row['id'],
                        'execution_id': row['id'], 'idea_id': row['idea_id'],
                        'project_epoch': row['project_epoch'], 'depends_on': [row['id']],
                        'status': 'evaluation_unavailable', 'error_type': type(exc).__name__,
                        'reason': str(exc)[:300], 'epistemic': 'GENERATED',
                        'scientific_support': 'not_assessed', 'mechanism_support': 'not_assessed'}
                events.append(make_event('ArtifactAdded', evaluated, action_type=EXPLORE))
                if isinstance(spec.get('compare_to'), str):
                    from ...services.native_evaluation import compare_evaluations
                    events.append(make_event('ArtifactAdded', compare_evaluations(
                        state, spec['compare_to'], evaluated, spec.get('intervention')), action_type=EXPLORE))
            else:
                row['evaluation_gap'] = 'no_host_registered_native_evaluator'
        return CandidateResult(status=row['status'], events=events, extras={'measured_cost': {'api_calls': 0, 'tokens': 0,
                'execution_seconds': (row.get('run') or {}).get('seconds', 0)
                    + (row.get('measurement_audit') or {}).get('seconds', 0)},
                'scientific_verdict': 'not_assessed'})
