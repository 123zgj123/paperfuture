"""Frozen, host-selected evaluators for files made by open research programs.

Candidate and evaluator run in separate namespace labs. Only selected candidate
files enter the evaluator; its private assets never enter the solving sandbox.
A metric contract gives a score its scope, not a scientific causal explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from .experiments import canonical, file_digest, fingerprint


@dataclass(frozen=True)
class NativeEvaluator:
    name: str
    code: Path
    assets: Mapping[str, Path]
    submission_files: tuple[str, ...]
    metrics: Mapping[str, Mapping[str, Any]]
    scope: str
    source: Mapping[str, Any]
    submission_help: str
    timeout: float = 120
    observables: Mapping[str, str] = field(default_factory=dict)
    delivery_policy: Mapping[str, Any] | None = None


def _name(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value))


def bind_evaluator(workspace: Path, evaluator: NativeEvaluator) -> dict[str, Any]:
    """Host API only. Freeze implementation, private inputs and metric semantics."""
    if not _name(evaluator.name) or not evaluator.scope or not evaluator.source:
        raise ValueError('native_evaluator_requires_identity_scope_and_source')
    if not evaluator.submission_files or len(set(evaluator.submission_files)) != len(evaluator.submission_files):
        raise ValueError('native_evaluator_requires_unique_submission_files')
    if any(not _name(n) for n in (*evaluator.assets, *evaluator.submission_files)):
        raise ValueError('native_evaluator_uses_explicit_flat_filenames')
    if set(evaluator.assets) & set(evaluator.submission_files):
        raise ValueError('candidate_cannot_replace_private_evaluator_assets')
    if not evaluator.metrics or not 0 < evaluator.timeout <= 3600:
        raise ValueError('native_evaluator_requires_metrics_and_bounded_timeout')
    for name, schema in evaluator.metrics.items():
        if (not _name(name) or not all(schema.get(k) for k in ('unit', 'meaning', 'population'))
                or schema.get('direction') not in ('higher', 'lower', 'descriptive')
                or not all(type(schema.get(k)) in (int, float) and math.isfinite(schema[k])
                           for k in ('minimum', 'maximum')) or schema['minimum'] > schema['maximum']):
            raise ValueError('native_metric_semantics_must_be_explicit')
    if any(not _name(n) or not isinstance(v, str) or not v.strip()
           for n, v in evaluator.observables.items()):
        raise ValueError('native_observables_require_names_and_definitions')
    if evaluator.delivery_policy is not None:
        policy = evaluator.delivery_policy
        if (set(policy) - {'stop_on_acceptance', 'minimum_metrics', 'maximum_metrics'}
                or type(policy.get('stop_on_acceptance', False)) is not bool):
            raise ValueError('invalid_native_delivery_policy')
        for key in ('minimum_metrics', 'maximum_metrics'):
            limits = policy.get(key, {})
            if not isinstance(limits, Mapping) or any(
                name not in evaluator.metrics or type(value) not in (int, float)
                or not math.isfinite(value)
                or not evaluator.metrics[name]['minimum'] <= value <= evaluator.metrics[name]['maximum']
                for name, value in limits.items()):
                raise ValueError('invalid_native_delivery_metric_threshold')
        if any(low > policy.get('maximum_metrics', {}).get(name, float('inf'))
               for name, low in policy.get('minimum_metrics', {}).items()):
            raise ValueError('contradictory_native_delivery_thresholds')
    files = {n: Path(p) for n, p in evaluator.assets.items()}
    code = Path(evaluator.code)
    for p in [code, *files.values()]:
        if p.is_symlink() or not p.is_file():
            raise ValueError('native_evaluator_requires_regular_files')
    manifest = {n: file_digest(p) for n, p in files.items()}
    public = {'name': evaluator.name, 'submission_files': list(evaluator.submission_files),
        'metrics': dict(evaluator.metrics), 'scope': evaluator.scope,
        'source': dict(evaluator.source), 'submission_help': evaluator.submission_help,
        'timeout': evaluator.timeout, 'clean_replay': 'same_program_fresh_directory_exact_output_bytes',
        'mechanism_support': 'not_assessed'}
    if evaluator.observables:
        public['observables'] = dict(evaluator.observables)
        public['observation_file'] = 'observations.json'
    if evaluator.delivery_policy is not None:
        public['delivery_policy'] = dict(evaluator.delivery_policy)
    config = {'public': public, 'code': code.read_text(), 'assets': manifest}
    identity = fingerprint(config)
    folder = Path(workspace) / 'native_evaluators' / identity
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'config.json'
    if not path.exists():
        with path.open('xb') as stream:
            stream.write(canonical(config))
    if json.loads(path.read_text()) != config:
        raise ValueError('native_evaluator_snapshot_changed')
    for name, source in files.items():
        target = folder / name
        if not target.exists():
            shutil.copyfile(source, target)
        if target.is_symlink() or file_digest(target) != manifest[name]:
            raise ValueError('native_evaluator_input_changed')
    return {'kind': 'native_evaluation_environment', 'id': identity,
            'contract_digest': identity, **public, 'epistemic': 'GENERATED'}


def capture_files(source: Path, names: list[str], destination: Path) -> dict[str, str]:
    """Copy only declared regular outputs, rejecting links and excessive size."""
    destination.mkdir(parents=True, exist_ok=False)
    total, manifest = 0, {}
    for name in names:
        if not _name(name):
            raise ValueError('invalid_submission_filename')
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing_or_linked_submission_file:' + name)
        total += path.stat().st_size
        if total > 64 * 1024 * 1024:
            raise ValueError('submission_size_limit')
        manifest[name] = file_digest(path)
        shutil.copyfile(path, destination / name)
        if file_digest(destination / name) != manifest[name]:
            raise ValueError('submission_changed_during_capture')
    return manifest


def evaluate_program(workspace: Path, contract: Mapping[str, Any], execution: Mapping[str, Any],
                     raw_root: Path, raw_names: list[str], run_folder: Path) -> dict[str, Any]:
    """Clean replay and independent scoring; all failures remain operational facts."""
    from ..scientist import lab
    identity = contract['contract_digest']
    root = Path(workspace) / 'native_evaluators' / identity
    config = json.loads((root / 'config.json').read_text())
    if fingerprint(config) != identity:
        raise ValueError('native_evaluator_contract_changed')
    for name, expected in config['assets'].items():
        if (root / name).is_symlink() or file_digest(root / name) != expected:
            raise ValueError('native_evaluator_private_input_changed')
    eid = 'evaluation-' + execution['id']
    folder = Path(workspace) / 'native_evaluations' / eid
    folder.mkdir(parents=True, exist_ok=False)
    result = {'kind': 'native_evaluation', 'id': eid, 'execution_id': execution['id'],
        'idea_id': execution['idea_id'], 'project_epoch': execution.get('project_epoch', 0),
        'depends_on': [execution['id']], 'contract_digest': identity,
        'scope': config['public']['scope'], 'metric_contract': config['public']['metrics'],
        'scientific_support': 'not_assessed', 'mechanism_support': 'not_assessed',
        'epistemic': 'GENERATED', 'status': 'started'}
    with (folder / 'started.json').open('xb') as stream:
        stream.write(canonical(result))
    try:
        names = config['public']['submission_files']
        observed = capture_files(run_folder / 'work', names, folder / 'candidate')
        result['candidate_files'] = observed
        spec = execution['proposal']
        replay = lab.run_once(spec['code'], spec.get('args', {}), workdir=folder / 'replay',
            timeout=contract['timeout'], data_root=raw_root, input_files=raw_names)
        result['replay'] = replay.to_dict()
        if not replay.ok:
            result.update(status='replay_failed', failure_class='execution_or_environment_failure')
        else:
            regenerated = capture_files(folder / 'replay', names, folder / 'frozen_submission')
            result['regenerated_files'] = regenerated
            result['submission_digest'] = fingerprint(regenerated)
            if observed != regenerated:
                result.update(status='replay_mismatch', failure_class='reproducibility_unresolved')
            else:
                inputs = folder / 'evaluation_inputs'
                inputs.mkdir()
                for name in names:
                    shutil.copyfile(folder / 'frozen_submission' / name, inputs / name)
                for name in config['assets']:
                    shutil.copyfile(root / name, inputs / name)
                measured = lab.run_once(config['code'], {}, workdir=folder / 'evaluator',
                    timeout=contract['timeout'], data_root=inputs,
                    input_files=sorted([*names, *config['assets']]))
                result['evaluator_run'] = measured.to_dict()
                if not measured.ok:
                    result.update(status='evaluator_failed', failure_class='measurement_or_evaluator_failure')
                else:
                    metrics = measured.result
                    schemas = config['public']['metrics']
                    valid = set(metrics) == set(schemas) and all(
                        schemas[n]['minimum'] <= value <= schemas[n]['maximum'] for n, value in metrics.items())
                    result.update(status='evaluated' if valid else 'metric_contract_failed', metrics=metrics,
                                  semantic_status='frozen_contract_scope_only' if valid else 'invalid')
                    if valid:
                        result['metric_feedback'] = {n: {'value': v, **schemas[n]} for n, v in metrics.items()}
                        # Only the independent evaluator writes these observations.
                        # Candidate-provided flags or code differences cannot stand
                        # in for a measured change in the targeted component.
                        if config['public'].get('observables'):
                            path = folder / 'evaluator' / config['public']['observation_file']
                            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
                                raise ValueError('missing_or_invalid_native_observations')
                            observed_components = json.loads(path.read_text())
                            if (not isinstance(observed_components, dict)
                                    or set(observed_components) != set(config['public']['observables'])
                                    or any(v is None for v in observed_components.values())):
                                raise ValueError('native_observation_schema_mismatch')
                            canonical(observed_components)  # rejects non-finite JSON
                            result['observable_measurements'] = observed_components
                            result['observable_definitions'] = config['public']['observables']
                            result['observation_file_digest'] = file_digest(path)
                    result['reported_metric_disagreements'] = [n for n, v in metrics.items()
                        if n in (execution.get('run') or {}).get('result', {}) and
                        not math.isclose(v, execution['run']['result'][n], rel_tol=1e-9, abs_tol=1e-12)]
    except (OSError, ValueError) as exc:
        result.update(status='evaluation_unavailable', failure_class='input_or_contract_failure',
                      error_type=type(exc).__name__, reason=str(exc)[:300])
    with (folder / 'result.json').open('xb') as stream:
        stream.write(canonical(result))
    return result


def compare_evaluations(state: Any, baseline_id: str, candidate: Mapping[str, Any],
                        intervention: Mapping[str, Any] | None = None) -> dict:
    """A paired score contrast, never a causal mechanism or population claim."""
    from ..extras.openworld.dependencies import dependency_view, resolve_reference
    from ..extras.openworld.inquiry import get
    baseline_id = resolve_reference(state, baseline_id)
    baseline = next((r for r in get(state, 'artifacts', [])
                     if r.get('kind') == 'native_evaluation' and r.get('id') == baseline_id), {})
    result = {'kind': 'native_comparison', 'id': 'comparison-' + candidate['id'],
        'idea_id': candidate['idea_id'], 'project_epoch': candidate.get('project_epoch', 0),
        'depends_on': [baseline_id, candidate['id']], 'status': 'not_comparable',
        'epistemic': 'GENERATED', 'mechanism_support': 'not_assessed',
        'population_inference': 'not_assessed'}
    if (baseline.get('status') != 'evaluated' or candidate.get('status') != 'evaluated'
            or baseline.get('contract_digest') != candidate.get('contract_digest')
            or dependency_view(state).get(baseline_id, {}).get('status') != 'current'):
        result['reason'] = 'requires_current_evaluations_under_identical_frozen_contract'
        return result
    result.update(status='descriptive_contrast',
        metric_delta={k: candidate['metrics'][k] - v for k, v in baseline['metrics'].items()},
        metric_contract=candidate['metric_contract'], scope=candidate['scope'])
    result['submission_changed'] = baseline.get('submission_digest') != candidate.get('submission_digest')
    if intervention is not None:
        result['manipulation_check'] = check_intervention(baseline, candidate, intervention)
    return result


def check_intervention(baseline: Mapping, candidate: Mapping, plan: Mapping) -> dict:
    """Check predeclared target and control observations, not causal explanation."""
    check = {'status': 'unassessed', 'mechanism_support': 'not_assessed'}
    if not isinstance(plan, Mapping):
        return {**check, 'reason': 'intervention_must_be_an_object'}
    targets, controls = plan.get('target_observables'), plan.get('invariant_observables')
    if (not isinstance(targets, list) or not targets or not isinstance(controls, list)
            or any(not isinstance(n, str) for n in targets + controls)
            or len(set(targets + controls)) != len(targets + controls)
            or not isinstance(plan.get('prediction'), str) or not plan['prediction'].strip()):
        return {**check, 'reason': 'requires_disjoint_targets_controls_and_prerun_prediction'}
    if (baseline.get('status') != 'evaluated' or candidate.get('status') != 'evaluated'
            or not baseline.get('contract_digest')
            or baseline['contract_digest'] != candidate.get('contract_digest')):
        return {**check, 'reason': 'requires_evaluations_under_identical_contract'}
    before, after = baseline.get('observable_measurements', {}), candidate.get('observable_measurements', {})
    if any(n not in before or n not in after for n in targets + controls):
        return {**check, 'reason': 'missing_independent_component_measurement'}
    changed = [n for n in targets if canonical(before[n]) != canonical(after[n])]
    confounded = [n for n in controls if canonical(before[n]) != canonical(after[n])]
    return {**check, 'status': 'confounded' if confounded else
            'observed_as_registered' if len(changed) == len(targets) else 'not_observed',
        'changed_targets': changed, 'unchanged_targets': [n for n in targets if n not in changed],
        'changed_controls': confounded, 'prediction': plan['prediction'],
        'scope': 'registered_observables_only_unmeasured_confounders_remain_possible'}
