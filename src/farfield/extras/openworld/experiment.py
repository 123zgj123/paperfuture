"""Small, explicit computational operators. No prose-to-template substitution.

The proposal names a supported operation and all of its parameters. The
registration is immutable across study stages. Unsupported proposals require
instrumentation; no fake inference score or arbitrary generated code is run.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

#: The largest edge set `connectivity_survival` will enumerate. The measure is
#: exact rather than sampled, so its cost is 2**|E|; past this the honest answer
#: is that the instrument does not reach, not a slower one.
SURVIVAL_EDGE_LIMIT = 16

OPERATIONS = {
    'graph_edit': ('simulation', ('mean_degree', 'edge_count',
                                  'components', 'connectivity_survival')),
    'repository_replace': ('repository', ('tests_passed',)),
    'table_classification': ('dataset', ('accuracy',)),
}
PROTOCOL_HELP = (
    'For an executable empirical idea also return experiment as a JSON object: '
    '{hypothesis: operational sentence for the chosen operator, never a copy of '
    'central_claim; operation, parameters, metric, '
    'expected_direction: treatment_higher|treatment_lower, margin: positive relative effect '
    'threshold, min_sample_size: positive integer}. Available bound-world interfaces appear '
    'under current_research_state.bound_world_apis.callable_operations: operation, observable/metric, '
    'parameters_schema and control_semantics describe the actual callable API. An admitted '
    'operation is executable even when it is not a built-in operator. For admitted operations '
    'parameters supplies the treatment inputs and parameters.control supplies inputs for the '
    'same operation in the control arm; omitted or empty control repeats the treatment inputs. '
    'An input schema inferred from public parameter access names does not declare input value '
    'domains or scientific outcomes. Built-in operator interfaces: '
    'graph_edit parameters {add_edges:[[a,b]], remove_edges:[[a,b]]}, metric mean_degree or '
    'edge_count, components (connected components) or connectivity_survival (the '
    'exact fraction of the 2**|E| edge subsets whose graph on all nodes stays '
    f'connected — all-terminal reliability at edge presence 1/2, enumerated, so it '
    'needs a world of at most 16 edges); repository_replace parameters '
    '{old:"old_text",new:"new_text"}, metric '
    'tests_passed; table_classification parameters {feature:"x",label:"y",test_fraction:0.3,seed:0}, '
    'metric accuracy (train-only nearest centroid versus train-only majority baseline on '
    'the same heldout rows). Graph samples count independent graphs, NOT nodes; repository '
    'samples count independent checkouts, NOT assertions. min_sample_size is a ceiling the '
    'bound world has to meet: it must not exceed independent_units_per_arm for the chosen '
    'operation under current_research_state.bound_world_apis. For graph_edit and '
    'repository_replace that number is 1, so registering more is refused as '
    'insufficient_independent_samples and nothing runs; coverage across a registered study '
    'does not raise it either. These are local fixture studies, '
    'not broad benchmark evidence. Unsupported experiments use experiment:{} and remain '
    'candidates. Do not change the research question to fit an operator. Every parameter, '
    'direction and metric must implement the proposed claim, not an unrelated demonstration.'
)
PROTOCOL_HELP += (
    ' A question about several settings at once takes a pre-registered study. Add '
    'experiment.study = {version:"multi-setting-study-v1", settings:[{id:"s0", '
    'parameters:{...}}, ...] (2-16, unique identifier ids, each overlaying the base '
    'parameters), unit_definition, aggregation:{method:"mean"|"every_setting", '
    'weights:"equal"}, stopping_rule:"all_registered_settings_once", '
    'missing_result_policy:"incomplete_until_every_setting_has_an_admitted_result", '
    'coverage_scope}. Every setting is then measured once, separately, before the study '
    'concludes anything, and the study is frozen: changing a setting, the aggregation or '
    'the scope is a new registration, not an edit. "every_setting" states a universal over '
    'the registered settings and needs each one to clear the margin; "mean" states only an '
    'average and establishes no individual setting. Settings are coverage, NOT sample size: '
    'five seeds on one bound world are still one independent unit, and min_sample_size still '
    'counts independent units per arm. With a study, experiment.hypothesis is the study '
    'sentence, which names the aggregation and the settings digest. Omit study for a '
    'single-setting comparison.'
)
PROTOCOL_HELP += (
    ' central_claim is the scientific hypothesis: mechanism, scope and competing '
    'explanations. It is NOT required to equal a fixture sentence. experiment.hypothesis '
    'is the operationalization: the exact comparison the operator will run. '
    'For the three built-in operators the operational sentence uses their controlled statement. '
    'An admitted operation uses the declared comparison of its actual input arms and observable. '
    'Graph: "Removing edges [] and adding edges [[0, 3]] raises mean_degree on the bound graph." '
    '(substitute edge lists, raises/lowers, and mean_degree/edge_count). Repository: '
    '"Replacing \'old_text\' with \'new_text\' raises tests_passed on the bound repository." '
    '(substitute registered values and raises/lowers). Table: '
    '"Nearest-centroid classification using x predicts y more accurately than the majority '
    'baseline on heldout rows (seed=0, test_fraction=0.3)." '
    '(substitute feature, label, more/less, seed and fraction). A fixture result supports '
    'only the operational sentence. It never automatically supports a broader scientific '
    'hypothesis. Keep unsupported experiments as experiment:{}.'
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def canonical_study_claim(p: Mapping[str, Any]) -> str:
    """The sentence a pre-registered multi-setting study is allowed to assert.

    It names the aggregation rule and carries the setting list's digest, so two
    studies over different settings never render the same sentence and cannot be
    mistaken for the same proposition. An average is stated as an average: it
    says nothing about any individual setting, and a universal has to be earned
    setting by setting.
    """
    from .study import settings_digest

    study = p['study']
    count = len(study['settings'])
    fingerprint = settings_digest(study)
    direction = 'raises' if p['expected_direction'] == 'treatment_higher' else 'lowers'
    if (study['aggregation'] or {}).get('method') == 'every_setting':
        return (f"In every one of the {count} registered settings (settings {fingerprint}), "
                f"the registered intervention {direction} {p['metric']} on the bound world.")
    return (f"Averaged with equal weight over the {count} registered settings "
            f"(settings {fingerprint}), the registered intervention {direction} "
            f"{p['metric']} on the bound world.")


def canonical_claim(p: Mapping[str, Any]) -> str:
    if isinstance(p.get('study'), Mapping):
        return canonical_study_claim(p)
    params = p['parameters']
    direction = 'raises' if p['expected_direction'] == 'treatment_higher' else 'lowers'
    if p['operation'] == 'graph_edit':
        return (f"Removing edges {json.dumps(params.get('remove_edges', []))} and adding edges "
                f"{json.dumps(params.get('add_edges', []))} {direction} {p['metric']} on the bound graph.")
    if p['operation'] == 'repository_replace':
        return (f"Replacing {params['old']!r} with {params['new']!r} {direction} "
                "tests_passed on the bound repository.")
    more = 'more' if p['expected_direction'] == 'treatment_higher' else 'less'
    return (f"Nearest-centroid classification using {params['feature']} predicts {params['label']} "
            f"{more} accurately than the majority baseline on heldout rows "
            f"(seed={params['seed']}, test_fraction={params['test_fraction']}).")


def protocol_errors(req: Any) -> list[str]:
    if str(getattr(req, 'kind', '') or '') == 'WORLD_EXTENSION':
        return ['world_extension_awaits_admission']
    p = req.execution
    if not isinstance(p, Mapping) or not p:
        return ['missing_executable_protocol']
    errors = []
    op = p.get('operation')
    builtin = isinstance(op, str) and op in OPERATIONS
    if not builtin:
        errors.append('unsupported_experiment_operator')
    metric = p.get('metric')
    if not isinstance(metric, str) or not metric.strip():
        errors.append('missing_registered_metric')
    elif req.primary_metric != metric:
        errors.append('operator_metric_or_world_mismatch')
    if builtin:
        klass, metrics = OPERATIONS[op]
        if req.world_class != klass or metric not in metrics:
            errors.append('operator_metric_or_world_mismatch')
    if not isinstance(p.get('hypothesis'), str) or not p['hypothesis'].strip():
        errors.append('missing_registered_hypothesis')
    if not isinstance(p.get('expected_direction'), str) or p['expected_direction'] not in {'treatment_higher', 'treatment_lower'}:
        errors.append('missing_registered_direction')
    margin = p.get('margin')
    try:
        valid_margin = (not isinstance(margin, bool) and isinstance(margin, (float, int))
                        and math.isfinite(margin) and margin > 0)
    except OverflowError:
        valid_margin = False
    if not valid_margin:
        errors.append('invalid_registered_margin')
    n = p.get('min_sample_size')
    if type(n) is not int or n <= 0:
        errors.append('invalid_registered_sample_floor')
    params = p.get('parameters')
    if not isinstance(params, Mapping):
        return errors + ['missing_operator_parameters']
    if not builtin:
        # The receipt can supply an implementation, never the missing study
        # definition. Keep every common registration error for fixture_errors.
        return errors
    if set(p) - {'hypothesis', 'operation', 'parameters', 'metric', 'expected_direction',
                 'margin', 'min_sample_size', 'study'}:
        errors.append('unsupported_protocol_parameter')
    errors.extend(parameter_errors(op, params))
    from .study import study_errors, setting_parameters, study_of
    if study_of(p) is not None:
        errors.extend(study_errors(p, lambda merged: parameter_errors(op, merged)))
    if not errors and p.get('hypothesis') != canonical_claim(p):
        errors.append('operational_claim_not_the_registered_operation')
    return errors


def parameter_errors(op: str, params: Mapping[str, Any]) -> list[str]:
    """Operator-level check of one parameter set — the base, or one setting.

    A study registers settings the operator could each have run alone, so both
    go through this same check rather than through two that could drift apart.
    """
    import re

    errors: list[str] = []
    allowed = {'graph_edit': {'add_edges', 'remove_edges'},
               'repository_replace': {'old', 'new'},
               'table_classification': {'feature', 'label', 'test_fraction', 'seed'}}.get(op)
    if allowed is None:
        return ['unsupported_experiment_operator']
    if set(params) - allowed:
        errors.append('unsupported_protocol_parameter')
    if op == 'graph_edit':
        for name in ('add_edges', 'remove_edges'):
            edges = params.get(name, [])
            if not isinstance(edges, list) or any(not isinstance(e, list) or len(e) != 2
                    or any(type(x) is not int for x in e) or e[0] == e[1] for e in edges):
                errors.append('invalid_graph_edit')
        if not params.get('add_edges') and not params.get('remove_edges'):
            errors.append('empty_graph_edit')
    elif op == 'repository_replace':
        # This operator supports the attested named-value fixture only. Arbitrary
        # code generation needs the existing registered/sandboxed worker path.
        if any(not isinstance(params.get(k), str) or not re.fullmatch(r'VALUE = -?\d{1,6}', params[k])
               for k in ('old', 'new')) or params.get('old') == params.get('new'):
            errors.append('unsupported_repository_patch')
    elif op == 'table_classification':
        if not all(isinstance(params.get(k), str) and params[k] for k in ('feature', 'label')):
            errors.append('missing_table_columns')
        if params.get('feature') == params.get('label'):
            errors.append('target_leakage')
        fraction = params.get('test_fraction')
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
            errors.append('invalid_test_split')
        if type(params.get('seed')) is not int:
            errors.append('missing_split_seed')
    return errors


_ALLOWED_AST = (
    ast.Module, ast.FunctionDef, ast.Return, ast.Assign, ast.If, ast.For,
    ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitXor, ast.BitAnd, ast.BitOr,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.And, ast.Or,
    ast.Not, ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Subscript, ast.Dict, ast.List, ast.Tuple, ast.Call, ast.Expr, ast.Pass,
    ast.arguments, ast.arg, ast.keyword,
)
_ALLOWED_CALLS = frozenset({"int", "abs", "min", "max", "len", "sum", "range", "sorted", "bool", "float"})


def implementation_errors(source: str) -> list[str]:
    """Restricted computational substrate. Not arbitrary Python authority."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["implementation_unparseable"]
    names: list[str] = []
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST:
            return [f"implementation_uses_forbidden_{type(node).__name__}"]
        if isinstance(node, (ast.Name, ast.arg, ast.FunctionDef)):
            name = node.id if isinstance(node, ast.Name) else node.arg if isinstance(node, ast.arg) else node.name
            if name.startswith('__'):
                return ['implementation_private_runtime_name_forbidden']
            binds = (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                     or isinstance(node, (ast.arg, ast.FunctionDef)))
            if binds and name in _ALLOWED_CALLS:
                return ['implementation_builtin_rebinding_forbidden']
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Name) or func.id not in _ALLOWED_CALLS:
                return ["implementation_call_not_on_allowlist"]
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
    if "act" not in names:
        return ["missing_act_function"]
    return []


def admitted_receipt(fixture: Any) -> dict[str, Any]:
    root = Path(getattr(fixture, "root", "") or "")
    path = root / "admitted_receipt.json"
    if not root or not path.is_file():
        return {}
    try:
        row = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return dict(row) if isinstance(row, Mapping) else {}


def independent_controls(fixture: Any) -> dict[str, Any]:
    root = Path(getattr(fixture, "root", "") or "")
    path = root / "independent_controls.json"
    if not root or not path.is_file():
        return {}
    try:
        row = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return dict(row) if isinstance(row, Mapping) else {}


def sandbox_act(source: str, params: Mapping[str, Any], dest: Path | None = None) -> dict[str, Any]:
    """Run a restricted act() in a subprocess. AST allowlist is not enough alone."""
    errors = implementation_errors(source)
    if errors:
        raise ValueError(';'.join(errors))
    serialized = json.dumps(params, allow_nan=False)
    if len(source.encode('utf-8')) > 256_000 or len(serialized.encode('utf-8')) > 16 * 1024**2:
        raise ValueError('sandbox_act_input_limit')
    if dest is None:
        with tempfile.TemporaryDirectory(prefix='farfield-act-') as temporary:
            return sandbox_act(source, params, Path(temporary))
    work = Path(dest)
    work.mkdir(parents=True, exist_ok=True)
    (work / 'impl.py').write_text(source)
    (work / 'params.json').write_text(serialized)
    (work / 'run.py').write_text(
        "import json, resource\nfrom pathlib import Path\n"
        "resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (5, 5))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "allowed = {" + ', '.join(f'{name!r}: {name}' for name in sorted(_ALLOWED_CALLS)) + "}\n"
        "scope = {'__builtins__': allowed}\n"
        "exec(compile(Path('impl.py').read_text(), 'impl.py', 'exec'), scope)\n"
        "out = scope['act'](json.loads(Path('params.json').read_text()))\n"
        "if not isinstance(out, dict):\n    raise SystemExit('act_must_return_dict')\n"
        "Path('out.json').write_text(json.dumps(out, allow_nan=False))\n"
    )
    env = {'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        result = subprocess.run([sys.executable, '-I', 'run.py'], cwd=work, env=env,
                                capture_output=True, timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('sandbox_act_timeout') from exc
    except OSError as exc:
        raise ValueError('sandbox_act_unavailable') from exc
    if result.returncode != 0 or not (work / 'out.json').is_file():
        raise ValueError('sandbox_act_failed')
    row = json.loads((work / 'out.json').read_text())
    if not isinstance(row, dict):
        raise ValueError('act_must_return_dict')
    return row


def run_independent_controls(source: str, controls: Mapping[str, Any],
                             dest: Path | None = None) -> dict[str, Any]:
    """Kernel acceptance. Proposer tests are not an input."""
    evaluator = str(controls.get('evaluator') or 'exact_match')
    if evaluator != 'exact_match':
        return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': 'unsupported_evaluator'}
    cases = list(controls.get('cases') or [])
    if not cases:
        return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': 'empty_independent_controls'}
    obs = str(controls.get('observable') or '')
    if not obs:
        return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': 'missing_control_observable'}
    outcomes: list[Any] = []
    root = Path(dest) if dest is not None else None
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or not isinstance(case.get('params'), Mapping):
            return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': 'invalid_control_case'}
        work = (root / f'case-{index}') if root is not None else None
        try:
            got = sandbox_act(source, case['params'], work)
        except ValueError as exc:
            return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': str(exc)}
        expect = case.get('expect') if isinstance(case.get('expect'), Mapping) else {}
        if obs not in got or got.get(obs) != expect.get(obs):
            return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': 'positive_control_failed'}
        outcomes.append(got)
    for index, case in enumerate(list(controls.get('negative') or [])):
        if not isinstance(case, Mapping) or not isinstance(case.get('params'), Mapping):
            return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': 'invalid_negative_case'}
        work = (root / f'neg-{index}') if root is not None else None
        try:
            got = sandbox_act(source, case['params'], work)
        except ValueError as exc:
            return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': str(exc)}
        forbid = case.get('forbid') if isinstance(case.get('forbid'), Mapping) else {}
        if obs in forbid and got.get(obs) == forbid.get(obs):
            return {'ok': False, 'status': 'INSTRUMENT_INVALID', 'reason': 'negative_control_failed'}
        outcomes.append(got)
    return {'ok': True, 'digest': digest(outcomes), 'outcomes': outcomes}


def _run_admitted_pair(root: Path, work: Path, receipt: Mapping[str, Any],
                       params: Mapping[str, Any]) -> tuple[float, float]:
    action = str(receipt.get('action') or '')
    source = (root / 'admitted' / f'{action}.py').read_text()
    obs = str(receipt.get('observable') or 'value')
    raw = dict(params or {})
    treatment_params = {key: value for key, value in raw.items() if key != 'control'}
    control_params = dict(raw.get('control') or {}) or dict(treatment_params)
    treatment = sandbox_act(source, treatment_params, work / 'treatment')
    control = sandbox_act(source, control_params, work / 'control')
    return float(treatment[obs]), float(control[obs])


def fixture_errors(req: Any, fixture: Any) -> list[str]:
    from .worldio import is_attested
    errors = protocol_errors(req)
    receipt = admitted_receipt(fixture)
    if errors == ['world_extension_awaits_admission']:
        wanted = str((getattr(req, 'extension', {}) or {}).get('action') or '')
        if receipt.get('action') and receipt.get('admitted') and receipt.get('action') == wanted:
            errors = []
        else:
            return errors
    if 'unsupported_experiment_operator' in errors:
        op = (req.execution or {}).get('operation')
        if receipt.get('action') == op and receipt.get('admitted'):
            errors.remove('unsupported_experiment_operator')
            observable = receipt.get('observable')
            if (not isinstance(observable, str) or not observable
                    or (req.execution or {}).get('metric') != observable
                    or req.primary_metric != observable):
                errors.append('admitted_observable_mismatch')
        else:
            return errors
    if errors:
        return errors
    if not is_attested(fixture):
        errors.append('unattested_or_changed_world_bytes')
    if fixture.schema != req.schema and not receipt.get('admitted'):
        errors.append('different_scientific_object_schema')
    if receipt.get('admitted') and (
            receipt.get('action') in {
                (req.execution or {}).get('operation'),
                (getattr(req, 'extension', {}) or {}).get('action')}):
        if 'admitted_receipt.json' not in set(fixture.files):
            errors.append('admitted_receipt_not_in_attested_manifest')
        return errors
    from .study import setting_ids, setting_parameters, study_of
    op = req.execution['operation']
    root = fixture.root
    consumed = {'graph_edit': {'graph.json'}, 'repository_replace': {'repo.py', 'test_repo.py'},
                'table_classification': {'table.json'}}[op]
    if not consumed <= set(fixture.files) or any(
            not (root / name).resolve().is_relative_to(root.resolve()) for name in consumed):
        return errors + ['consumed_payload_not_in_attested_manifest']
    # A study has to be applicable at every setting it registered, not on
    # average. One inapplicable setting is an unrunnable study.
    arms = ([(sid, setting_parameters(req.execution, sid)) for sid in setting_ids(req.execution)]
            if study_of(req.execution) is not None else [('', req.execution['parameters'])])
    for label, params in arms:
        errors.extend(f'{label}:{problem}' if label else problem
                      for problem in _arm_errors(req, fixture, op, params))
    return errors


def _arm_errors(req: Any, fixture: Any, op: str, params: Mapping[str, Any]) -> list[str]:
    """Can this operator actually run these parameters on this frozen world?"""
    errors: list[str] = []
    root = fixture.root
    try:
        if op == 'graph_edit':
            graph = json.loads((root / 'graph.json').read_text())
            nodes = set(graph['nodes'])
            edges = {tuple(sorted(e)) for e in graph['edges']}
            if not nodes or len(nodes) != len(graph['nodes']) or len(edges) != len(graph['edges']) or any(
                    len(e) != 2 or e[0] == e[1] or not set(e) <= nodes for e in graph['edges']):
                errors.append('invalid_simple_graph')
            adds = {tuple(sorted(e)) for e in params.get('add_edges', [])}
            removes = {tuple(sorted(e)) for e in params.get('remove_edges', [])}
            if adds & edges or not removes <= edges or adds & removes or any(not set(e) <= nodes for e in adds | removes):
                errors.append('graph_edit_not_applicable')
            # Survival is enumerated, not sampled, so its reach is a property of
            # the world rather than of the run. Refuse it at registration where
            # it can be reported, instead of failing mid-execution.
            if req.execution.get('metric') == 'connectivity_survival':
                widest = max(len(edges), len((edges - removes) | adds))
                if widest > SURVIVAL_EDGE_LIMIT:
                    errors.append('connectivity_survival_needs_at_most_'
                                  f'{SURVIVAL_EDGE_LIMIT}_edges_but_this_world_has_{widest}')
            samples = 1
        elif op == 'repository_replace':
            code = (root / 'repo.py').read_text()
            tests = (root / 'test_repo.py').read_text()
            expected = ("import unittest\nimport repo\nclass T(unittest.TestCase):\n"
                        "    def test_score(self):\n        self.assertEqual(repo.score(), 2)\n")
            if code != params['old'] + '\ndef score():\n    return VALUE\n' or tests != expected:
                errors.append('repository_requires_registered_sandbox_executor')
            samples = 1
        else:
            rows = json.loads((root / 'table.json').read_text())['rows']
            if len(rows) < 4 or any(not isinstance(r.get(params['feature']), (int, float))
                    or not math.isfinite(r[params['feature']]) or params['label'] not in r for r in rows):
                errors.append('invalid_numeric_table')
            samples = max(1, int(len(rows) * params['test_fraction']))
        if samples < req.minimum_sample_or_task_count:
            # A registered floor above what one arm can supply. Coverage across
            # settings is a different count and never fills this one. Name both
            # numbers: "insufficient" alone told a proposer nothing it could act
            # on, and it registered the same impossible floor three times.
            errors.append('insufficient_independent_samples:'
                          f'registered_{req.minimum_sample_or_task_count}'
                          f'_but_this_world_supplies_{samples}_per_arm')
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        errors.append('missing_or_invalid_operator_payload')
    return errors


def safe_identifier(value: Any) -> bool:
    """Identifiers are path components, never paths supplied by a worker."""
    return (isinstance(value, str) and bool(value) and value not in {'.', '..'}
            and '/' not in value and '\\' not in value and '\x00' not in value)


def confined_path(root: Path, *parts: str) -> Path:
    root = Path(root).resolve()
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root) or any(
            parent.is_symlink() for parent in (path, *path.parents) if parent != root):
        raise ValueError('computational_path_escapes_workspace')
    return path


def computational_theory(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Kernel-only projection of an admitted operational claim, never mechanism proof."""
    reg = row.get('registration') or {}
    if (row.get('epistemic') != 'WORLD' or not row.get('attested')
            or row.get('invalidated') or not reg.get('claim_id')
            or row.get('theory_id') != reg.get('claim_id')):
        return None
    return {'id': reg['claim_id'], 'question_id': row.get('question_id', ''),
            'discovery_idea_id': row.get('discovery_idea_id', ''),
            'text': reg['hypothesis'], 'claim': reg['hypothesis'],
            'scientific_hypothesis': reg.get('scientific_hypothesis', ''),
            'evidence_scope': list((row.get('card') or {}).get('claim_spec', {}).get('evidence_scope') or []),
            'epistemic': 'GENERATED', 'role': 'operational_claim'}


def adaptive_criterion_sources(execution: Mapping[str, Any], state: Any) -> list[dict[str, Any]]:
    """Observed comparisons whose decision criteria this proposal changes.

    Renaming an idea or retiring a lineage does not make observed data unseen.
    This records selection history, not a prohibition on revising hypotheses.
    """
    rows = state.get('evidence_records', []) if isinstance(state, Mapping) else state.evidence_records
    sources = []
    for row in rows:
        prior = (row.get('registration') or {}).get('execution') or {}
        if (not prior or row.get('invalidated') or row.get('epistemic') != 'WORLD'
                or not row.get('attested') or row.get('execution_status') != 'ran'):
            continue
        # Changing an intervention, graph edge spelling, or split seed cannot
        # make already inspected data eligible for a relaxed decision rule.
        if (all(execution.get(k) == prior.get(k) for k in ('operation', 'metric'))
                and any(execution.get(k) != prior.get(k)
                        for k in ('margin', 'expected_direction', 'min_sample_size'))):
            sources.append(row)
    return sources


def adaptive_world_error(execution: Mapping[str, Any], state: Any,
                         fixture: Any, workspace: Path, evidence_id: str = '') -> str:
    """Revised criteria need fresh data, not another name for the old fixture."""
    from ..world import load_fixture
    # A replay must use the selection history at the original admission, not
    # retroactively invalidate an older measurement because of a later study.
    rows = state.get('evidence_records', []) if isinstance(state, Mapping) else state.evidence_records
    versions = state.get('world_versions', []) if isinstance(state, Mapping) else state.world_versions
    if evidence_id:
        index = next((i for i, r in enumerate(rows) if r.get('evidence_id') == evidence_id), len(rows))
        state = {'evidence_records': rows[:index]}
    sources = adaptive_criterion_sources(execution, state)
    for row in sources:
        if row.get('world_digest') == fixture.digest:
            return 'adaptive_criteria_reuse_observed_world'
        try:
            previous = load_fixture(confined_path(workspace, 'worlds', row['world_id']))
            if scientific_payload_digest(previous, execution['operation'], execution['parameters']) == scientific_payload_digest(
                    fixture, execution['operation'], execution['parameters']):
                return 'adaptive_criteria_reuse_observed_world'
        except (OSError, ValueError, KeyError, TypeError):
            return 'adaptive_criteria_independence_unverifiable'
    if sources:
        if not any(w.get('id') == fixture.id and w.get('research_role') == 'heldout'
                   and w.get('digest') == fixture.digest and w.get('attested') for w in versions):
            return 'adaptive_criteria_require_declared_heldout_world'
        # The revised registration may continue its own pilot, but a different
        # prior comparison on the evaluation payload makes it already inspected.
        history = state.get('evidence_records', []) if isinstance(state, Mapping) else state.evidence_records
        for row in history:
            prior = (row.get('registration') or {}).get('execution') or {}
            if (prior == execution or row.get('epistemic') != 'WORLD'
                    or not row.get('attested') or row.get('execution_status') != 'ran'):
                continue
            if row.get('world_digest') == fixture.digest:
                return 'adaptive_criteria_reuse_observed_world'
            if prior.get('operation') != execution.get('operation'):
                continue
            try:
                observed = load_fixture(confined_path(workspace, 'worlds', row['world_id']))
                if scientific_payload_digest(observed, execution['operation'], execution['parameters']) == scientific_payload_digest(
                        fixture, execution['operation'], execution['parameters']):
                    return 'adaptive_criteria_reuse_observed_world'
            except (OSError, ValueError, KeyError, TypeError):
                return 'adaptive_criteria_independence_unverifiable'
    return ''


def registration(req: Any) -> dict[str, Any]:
    execution = req.execution or {}
    operational = execution.get('hypothesis') or (
        canonical_claim(execution) if execution.get('operation') in OPERATIONS else '')
    # A study is a different protocol, so it says so. Single-setting
    # registrations keep the v3 identity byte for byte: their historical digests
    # stay valid, and nothing already recorded is re-signed as something newer.
    from .study import STUDY_VERSION, study_of
    version = ('computational-protocol-v4' if study_of(execution) is not None
               else 'computational-protocol-v3')
    return {'version': version, 'claim_id': req.claim_id,
            'question_id': req.question_id,
            'hypothesis': operational, 'scientific_hypothesis': req.hypothesis,
            'scientific_definition': {
                'estimand': 'registered treatment minus control metric on the frozen world',
                'decision_statistic': '(treatment-control)/abs(control) when control != 0; treatment-control otherwise',
                'decision_rule': 'expected-direction statistic >= margin corroborates; <= -margin refutes; otherwise inconclusive',
                'population': 'only the world_id and world_digest in the enclosing protocol; no population extrapolation',
                'intervention': {'operation': execution.get('operation'), 'parameters': execution.get('parameters')},
                'comparator': 'the registered operator control on the same frozen world and evaluation units',
                'observable': req.measurement,
                'metric': execution.get('metric'),
                'expected_direction': execution.get('expected_direction'),
                'decision_margin': execution.get('margin'),
                'study': dict(study_of(execution) or {}) or None,
                'study_version': STUDY_VERSION if study_of(execution) is not None else '',
                'required_assumptions': list(req.major_confounders or ()),
                'declared_requirement': req.to_dict(),
                'sufficiency': 'not_established_by_registration_or_minimum_sample_floor',
            },
            'execution': execution,
            'minimum_sample_or_task_count': req.minimum_sample_or_task_count,
            'major_confounders': list(req.major_confounders or ()),
            'engine_digest': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def inference_link(req: Any, *, study_stage: str, relation: str) -> dict[str, Any]:
    """Evidence supports the operational claim only. Never a silent leap."""
    execution = req.execution or {}
    operational = execution.get('hypothesis') or (
        canonical_claim(execution) if execution.get('operation') in OPERATIONS else '')
    same = (req.hypothesis or '').strip() == (operational or '').strip()
    # Instrument admission verifies code against independent control outputs;
    # it does not establish that an arbitrary prose label describes those
    # outputs. Only a kernel-defined operational statement has that mapping.
    mapped = (execution.get('operation') in OPERATIONS
              and operational == canonical_claim(execution))
    inherit = bool(mapped and same and study_stage in {'confirmatory', 'replication'}
                   and relation in {'supports', 'contradict'})
    return {
        'from_object': 'evidence',
        'operational_claim': operational,
        'scientific_hypothesis': req.hypothesis,
        'relation': relation,
        'scope': 'registered_operation_on_attested_world',
        'assumptions': list(req.major_confounders or ()),
        'possible_confounders': list(req.major_confounders or ()),
        'inherits_to_scientific_hypothesis': inherit,
        'note': ('' if inherit else
                 'fixture or cheap-run result does not automatically support the scientific hypothesis'),
    }


def scientific_payload_digest(fixture: Any, operation: str, parameters: Mapping[str, Any] | None = None) -> str:
    """Ignore names, manifest metadata, formatting and row/edge ordering."""
    if operation == 'graph_edit':
        g = json.loads((fixture.root / 'graph.json').read_text())
        return digest({'nodes': sorted(g['nodes']), 'edges': sorted(sorted(e) for e in g['edges'])})
    if operation == 'table_classification':
        table = json.loads((fixture.root / 'table.json').read_text())
        if not parameters:
            raise ValueError('replication_requires_registered_columns')
        return digest(sorted(digest([r[parameters['feature']], r[parameters['label']]]) for r in table['rows']))
    if operation == 'repository_replace':
        import ast
        return digest([ast.dump(ast.parse((fixture.root / name).read_text()))
                       for name in ('repo.py', 'test_repo.py')])
    receipt = admitted_receipt(fixture)
    action = str(receipt.get('action') or operation or '')
    source_path = Path(getattr(fixture, 'root', '') or '') / 'admitted' / f'{action}.py'
    if receipt.get('admitted') and source_path.is_file():
        return digest({'action': action, 'implementation': source_path.read_text()})
    raise ValueError('unsupported_replication_operator')


def _components(nodes: list[Any], edges: Any) -> int:
    """Connected components of the graph on ``nodes`` with exactly ``edges``."""
    adjacency: dict[Any, set[Any]] = {node: set() for node in nodes}
    for left, right in edges:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen: set[Any] = set()
    count = 0
    for node in nodes:
        if node in seen:
            continue
        count += 1
        stack = [node]
        seen.add(node)
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return count


def _connectivity_survival(nodes: list[Any], edges: Any) -> float:
    """All-terminal reliability at independent edge presence 1/2, computed exactly.

    The fraction of the 2**|E| edge subsets whose graph on *all* nodes is
    connected. Two live missions asked for exactly this — whether a chord makes
    a graph hold together under random edge failure — and the project could only
    offer mean degree, which answers a different question.

    It is enumerated rather than sampled, so the number is exact and replays
    bit-for-bit. That also fixes its cost at 2**|E|, so beyond
    ``SURVIVAL_EDGE_LIMIT`` it refuses instead of running slowly: an instrument
    that does not reach should say so.
    """
    ordered = sorted(tuple(sorted(edge)) for edge in edges)
    if len(ordered) > SURVIVAL_EDGE_LIMIT:
        raise ValueError('connectivity_survival_exceeds_enumerable_edge_count')
    if len(nodes) == 1:
        return 1.0
    connected = 0
    for mask in range(1 << len(ordered)):
        present = [edge for index, edge in enumerate(ordered) if mask >> index & 1]
        if _components(nodes, present) == 1:
            connected += 1
    return connected / float(1 << len(ordered))


def execute_protocol(protocol: Mapping[str, Any], root: Path, work: Path,
                     *, setting_id: str = "") -> dict[str, Any]:
    """Execute through the shared receipt service; retain the numeric result API."""
    from ...services import experiments

    root, work = Path(root), Path(work)
    # The world has already been admitted by the caller. Bind all files in the
    # supplied frozen directory, including admitted implementation/control code.
    files = {str(path.relative_to(root)): experiments.file_digest(path)
             for path in sorted(root.rglob("*")) if path.is_file()}
    frozen = json.loads(experiments.canonical(dict(protocol)))
    spec = {"backend": "openworld.computational", "backend_digest": experiments.file_digest(Path(__file__)),
            "protocol": frozen, "setting_id": setting_id, "input_files": files,
            "dependency_environment_attested": False}

    def run() -> dict[str, Any]:
        result = _execute_protocol(frozen, root, work, setting_id=setting_id)
        after = {str(path.relative_to(root)): experiments.file_digest(path)
                 for path in sorted(root.rglob("*")) if path.is_file()}
        if after != files:
            raise ValueError("experiment_inputs_changed")
        return result

    return experiments.execute(spec, records=work.with_name(work.name + ".experiments"),
        backend=run, summarize=lambda value: value, succeeded=lambda value: True).value


def _execute_protocol(protocol: Mapping[str, Any], root: Path, work: Path,
                     *, setting_id: str = "") -> dict[str, Any]:
    """Execute only the frozen structured intervention. No model judgement.

    ``setting_id`` selects one registered setting of a pre-registered study. The
    parameters it runs with are the protocol's own, overlaid by that setting, so
    a setting can only vary what the study registered it to vary. An unknown id
    is a refusal, never a silent fall back to the base parameters.
    """
    from .study import setting_ids, setting_parameters, study_of

    p = protocol['execution']; params = p['parameters']; op = p['operation']
    if setting_id:
        if study_of(p) is None or setting_id not in setting_ids(p):
            raise ValueError('unregistered_study_setting')
        params = setting_parameters(p, setting_id)
    elif study_of(p) is not None:
        raise ValueError('study_requires_a_registered_setting')
    if op == 'graph_edit':
        g = json.loads((root / 'graph.json').read_text())
        control_edges = {tuple(sorted(e)) for e in g['edges']}
        treatment_edges = (control_edges - {tuple(sorted(e)) for e in params.get('remove_edges', [])}) | {
            tuple(sorted(e)) for e in params.get('add_edges', [])}
        nodes = list(g['nodes'])

        def metric(edges):
            name = p['metric']
            if name == 'edge_count':
                return float(len(edges))
            if name == 'mean_degree':
                return 2 * len(edges) / len(nodes)
            if name == 'components':
                return float(_components(nodes, edges))
            return _connectivity_survival(nodes, edges)
        treatment, control = metric(treatment_edges), metric(control_edges)
        n, basis = 1, 'one_independent_graph_fixture_per_arm; nodes_are_not_replicates'
    elif op == 'repository_replace':
        import shutil
        work.mkdir(exist_ok=True)
        for name in ('repo.py', 'test_repo.py'):
            shutil.copyfile(root / name, work / name)
        # Deliberately pass only operational environment, never provider secrets.
        env = {'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1'}
        def check():
            result = subprocess.run([sys.executable, '-m', 'unittest', 'test_repo'], cwd=work,
                                    env=env, capture_output=True, timeout=10)
            if result.returncode not in (0, 1):
                raise ValueError('test_process_failed')
            if b'Ran 1 test' not in result.stderr or b'ERROR:' in result.stderr:
                raise ValueError('test_did_not_execute')
            return float(result.returncode == 0)
        control = check()
        code = (work / 'repo.py').read_text()
        (work / 'repo.py').write_text(code.replace(params['old'], params['new'], 1))
        treatment = check()
        n, basis = 1, 'one_independent_repository_checkout_per_arm'
    elif op == 'table_classification':
        rows = json.loads((root / 'table.json').read_text())['rows']
        indexes = list(range(len(rows))); random.Random(params['seed']).shuffle(indexes)
        n = max(1, int(len(rows) * params['test_fraction']))
        train, test = [rows[i] for i in indexes[n:]], [rows[i] for i in indexes[:n]]
        if not train or not test:
            raise ValueError('empty_train_or_test')
        feature, label = params['feature'], params['label']
        labels = sorted({r[label] for r in train}, key=str)
        centroids = {c: sum(r[feature] for r in train if r[label] == c) /
                     sum(r[label] == c for r in train) for c in labels}
        majority = max(labels, key=lambda c: sum(r[label] == c for r in train))
        treatment = sum(min(labels, key=lambda c: abs(r[feature] - centroids[c])) == r[label] for r in test) / n
        control = sum(majority == r[label] for r in test) / n
        basis = 'heldout_rows_disjoint_from_training; paired_arms_on_identical_test_rows'
    else:
        receipt = admitted_receipt(type('F', (), {'root': root, 'files': ()})())
        if not receipt:
            try:
                receipt = json.loads((root / 'admitted_receipt.json').read_text())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError('unsupported_experiment_operator') from exc
        if receipt.get('action') != op or not receipt.get('admitted'):
            raise ValueError('unsupported_experiment_operator')
        treatment, control = _run_admitted_pair(root, work, receipt, params)
        n, basis = 1, 'admitted_action_on_attested_world; independent_controls_are_not_this_result'
    return {'treatment': treatment, 'control': control, 'n_treatment': n, 'n_control': n,
            'sample_size': n, 'sample_size_basis': basis, 'primary_metric': p['metric']}


def claim_spec(req: Any, fixture: Any) -> dict[str, Any]:
    from ..claimspec import ClaimSpec, ClaimType, HandleSpec, HandleKind, MeasurementSpec, MechanismSpec
    op = req.execution['operation']
    handle = op + ':' + digest(req.execution['parameters'])[:16]
    return ClaimSpec(claim_id=req.claim_id, target_object=fixture.id,
        mechanism=MechanismSpec(name=req.intervention),
        handle=HandleSpec(id=handle, kind=HandleKind.INTERVENTIONAL, world_refs=(fixture.id,)),
        measurement=MeasurementSpec(dv=req.primary_metric, source_fields=(req.primary_metric,)),
        claim_type=ClaimType.PREDICTIVE if op == 'table_classification' else ClaimType.INTERVENTION,
        statement=(canonical_claim(req.execution) if op in OPERATIONS
                   else str(req.execution.get('hypothesis') or '')),
        evidence_scope=(f'fixed_protocol({handle},{req.primary_metric},world={fixture.id},digest={fixture.digest})',)
    ).to_dict()


def admission_errors(row: Mapping[str, Any], workspace: Path | None, state: Any) -> list[str]:
    """Check computational receipts at the existing trusted admission boundary."""
    if not row.get('registration'):
        return ['missing_computational_registration']
    from ..world import digest_files, load_fixture
    from .worldio import is_attested, replication_compatible
    from .requirement import _pilot_script
    from .study import setting_ids, study_of
    from ..validity import evidence_eligible, corroborates
    try:
        reg = row['registration']; cid = reg['claim_id']; stage = row['study_stage']
        if not workspace or not safe_identifier(cid):
            return ['invalid_computational_claim_id']
        label = {'pilot': 'pilot', 'scientific_pilot': 'pilot', 'calibration': 'calibrate',
                 'confirmatory': 'confirm', 'replication': 'replicate'}[stage]
        setting_id = str(row.get('study_setting_id') or '')
        if setting_id and not safe_identifier(setting_id):
            return ['invalid_study_setting_id']
        folder = f'{label}-{cid}-{setting_id}' if setting_id else f'{label}-{cid}'
        dest = confined_path(Path(workspace), 'research_workers', folder)
        protocol = json.loads((dest / 'protocol.json').read_text())
        metrics = json.loads((dest / 'metrics.json').read_text())
        wid = row['world_id']
        if not safe_identifier(wid):
            return ['invalid_computational_world_id']
        fixture = load_fixture(confined_path(Path(workspace), 'worlds', wid))
        if (not is_attested(fixture) or protocol['world_id'] != wid
                or protocol['world_digest'] != fixture.digest or row['world_digest'] != fixture.digest
                or protocol['study_stage'] != stage or protocol['registration'] != reg
                or str(protocol.get('study_setting_id') or '') != setting_id
                or (setting_id and setting_id not in setting_ids(reg['execution']))
                or (not setting_id and study_of(reg['execution']) is not None)
                or row.get('discovery_idea_id') != cid
                or digest(reg) != row.get('registration_digest')
                or digest(reg) != protocol['registration_digest']
                or reg['engine_digest'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or digest_files(dest / 'data', fixture.files) != fixture.digest
                or (dest / 'experiment.py').read_text() != _pilot_script('')):
            return ['computational_receipt_identity_mismatch']
        adaptive_error = adaptive_world_error(reg['execution'], state, fixture, Path(workspace),
                                               str(row.get('evidence_id') or ''))
        if adaptive_error:
            return [adaptive_error]
        for name in ('treatment', 'control', 'sample_size', 'sample_size_basis', 'primary_metric'):
            if row.get(name) != metrics.get(name):
                return ['computational_metrics_mismatch']
        if row.get('margin') != reg['execution']['margin'] or row.get('expected_direction') != reg['execution']['expected_direction']:
            return ['computational_judgement_registration_mismatch']
        from .requirement import ExperimentRequirement
        req = ExperimentRequirement(claim_id=cid, claim_type='',
                hypothesis=reg.get('scientific_hypothesis', ''),
                major_confounders=tuple(reg.get('major_confounders') or ()),
                falsifier='', execution=reg['execution'], primary_metric=reg['execution']['metric'])
        from .requirement import judge_pilot
        pilot_verdict = judge_pilot(req, metrics)
        relation = {'PILOT_POSITIVE': 'supports', 'PILOT_NEGATIVE': 'contradict'}.get(pilot_verdict, 'inconclusive')
        if (row.get('theory_id') != cid
                or row.get('inference_link') != inference_link(req, study_stage=stage, relation=relation)):
            return ['computational_inference_binding_mismatch']
        for theory in state.theories:
            if theory.get('id') == cid and (theory.get('claim') or theory.get('text')) != reg['hypothesis']:
                return ['computational_theory_identity_collision']
        if row.get('pilot_verdict') != pilot_verdict:
            return ['computational_stage_verdict_mismatch']
        if stage in {'confirmatory', 'replication'} and row.get('stage_verdict') != {
                'PILOT_POSITIVE': 'POSITIVE', 'PILOT_NEGATIVE': 'NEGATIVE'}.get(pilot_verdict, 'INCONCLUSIVE'):
            return ['computational_stage_verdict_mismatch']
        expected_claim = claim_spec(req, fixture)
        recorded = row.get('card', {}).get('claim_spec', {})
        if any(recorded.get(k) != expected_claim.get(k) for k in (
                'claim_id', 'target_object', 'handle', 'measurement', 'claim_type', 'statement', 'evidence_scope')):
            return ['computational_claim_spec_mismatch']
        previous = [r for r in state.evidence_records if r.get('registration_digest') == digest(reg)
                    and r.get('discovery_idea_id') == cid and evidence_eligible(r)]
        if stage == 'confirmatory' and not any(r.get('study_stage') in {'pilot', 'scientific_pilot'}
                and (r.get('pilot_verdict') == 'PILOT_POSITIVE' or (
                    r.get('pilot_verdict') == 'PILOT_NEGATIVE' and r.get('diagnosed') and r.get('replay_ok')))
                and r.get('world_digest') == fixture.digest
                for r in previous):
            return ['confirmation_without_registered_positive_scientific_pilot']
        if stage == 'replication':
            prior = next((r for r in previous if r.get('evidence_id') == row.get('replication_of')
                          and r.get('study_stage') == 'confirmatory'
                          and corroborates(r)
                          and r.get('stage_verdict') == 'POSITIVE'), None)
            if not prior:
                return ['replication_without_confirmatory_reference']
            primary = load_fixture(confined_path(Path(workspace), 'worlds', prior['world_id']))
            reason = replication_compatible(primary, fixture)
            if reason or primary.digest != prior.get('world_digest'):
                return ['replication_identity_or_independence_failure']
            if scientific_payload_digest(primary, reg['execution']['operation'], reg['execution']['parameters']) == scientific_payload_digest(fixture, reg['execution']['operation'], reg['execution']['parameters']):
                return ['replication_reuses_scientific_payload']
            if not any(w.get('id') == wid and w.get('research_role') == 'heldout' for w in state.world_versions):
                return ['replication_not_declared_heldout']
            if (row.get('independent_replication') is not True
                    or row.get('transfer_ok') != (pilot_verdict == 'PILOT_POSITIVE')):
                return ['computational_replication_verdict_mismatch']
        elif row.get('replication_of') or row.get('independent_replication') or row.get('transfer_ok'):
            return ['computational_replication_stage_mismatch']
    except (OSError, ValueError, KeyError, TypeError):
        return ['missing_or_invalid_computational_receipt']
    return []
