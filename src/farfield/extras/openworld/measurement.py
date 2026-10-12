"""Host-owned, bounded table checks; not a scientific or semantic evaluator.

Never infer a header, independent unit, biological label or causal mechanism.
Bindings express the researcher's measurement definition; passing verifies the
arithmetic conditional on that definition, not the definition's appropriateness.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import math
from pathlib import Path
import time
from typing import Any, Mapping

from ...services.experiments import file_digest, fingerprint

VERSION = 'table-measurement-v1'
MAX_BYTES = 256 * 1024 * 1024
MAX_LINE = 1024 * 1024
MAX_ROWS = 2_000_000


def profile_table(path: Path, *, max_bytes: int = MAX_BYTES,
                  max_rows: int = MAX_ROWS, seconds: float = 15) -> dict[str, Any]:
    """Read CSV logical records without strip(), header inference or data repair.

    Completion only means this parser reached EOF. UTF-8, delimiter and CSV
    quoting are explicit conventions, not evidence of original data integrity.
    """
    suffix = path.name.removesuffix('.gz')
    if not suffix.endswith(('.csv', '.tsv')):
        return {'status': 'unsupported', 'reason': 'no_registered_table_parser'}
    delimiter = '\t' if suffix.endswith('.tsv') else ','
    out: dict[str, Any] = {'status': 'incomplete', 'delimiter': delimiter,
        'encoding': 'utf-8', 'header_inferred': False, 'records': 0,
        'nonempty_records': 0, 'first_records': []}
    widths: Counter[int] = Counter()
    started, consumed = time.monotonic(), 0
    opener = gzip.open if path.name.endswith('.gz') else open

    def lines(handle):
        nonlocal consumed
        while True:
            if time.monotonic() - started > seconds:
                raise ValueError('time_limit')
            line = handle.readline(MAX_LINE + 1)
            if not line:
                return
            consumed += len(line)
            if consumed > max_bytes or len(line) > MAX_LINE:
                raise ValueError('byte_or_line_limit')
            yield line.decode('utf-8')

    try:
        with opener(path, 'rb') as stream:
            for row in csv.reader(lines(stream), delimiter=delimiter, strict=True):
                if out['records'] >= max_rows:
                    raise ValueError('record_limit')
                out['records'] += 1
                out['nonempty_records'] += bool(row and any(cell.strip() for cell in row))
                widths[len(row)] += 1
                if len(out['first_records']) < 5:
                    types = []
                    for cell in row[:32]:
                        try:
                            kind = 'number' if math.isfinite(float(cell)) else 'nonfinite'
                        except ValueError:
                            kind = 'text' if cell else 'empty'
                        types.append(kind)
                    out['first_records'].append({'fields': len(row), 'types': types})
        out['status'] = 'complete'
    except (OSError, EOFError, UnicodeError, csv.Error, ValueError) as exc:
        out['reason'] = str(exc)[:160]
    out['field_counts'] = {str(k): v for k, v in sorted(widths.items())}
    out['decoded_bytes'] = consumed
    return out


def audit_measurements(root: Path, assets: Mapping[str, Any], proposal: Mapping[str, Any],
                       metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Audit raw structure and explicitly bound numeric outputs independently.

    Arbitrary code is never run by this checker. Unbound metrics remain
    unverified. Asset hashes and checker version bind each result to its inputs.
    """
    started = time.monotonic()
    profiles = {}
    for name, expected in assets.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or file_digest(path) != expected['sha256']:
            profiles[name] = {'status': 'input_changed'}
        else:
            profiles[name] = profile_table(path, seconds=max(0, min(15,
                30 - (time.monotonic() - started))))
            if file_digest(path) != expected['sha256']:
                profiles[name] = {'status': 'input_changed'}
    bindings = proposal.get('measurement_bindings', [])
    checks = []
    if not isinstance(bindings, list) or len(bindings) > 64:
        bindings = [{}]
    for index, binding in enumerate(bindings):
        check = {'check_id': f'measurement-{index}', 'status': 'unverified'}
        checks.append(check)
        if not isinstance(binding, Mapping):
            check['reason'] = 'invalid_binding'
            continue
        check['definition'] = dict(binding)
        metric, asset = binding.get('metric'), binding.get('asset')
        if not isinstance(metric, str) or not isinstance(asset, str):
            check['reason'] = 'metric_and_asset_required'
            continue
        check['metric'] = metric
        profile = profiles.get(asset, {})
        if profile.get('status') != 'complete':
            check['reason'] = 'complete_independent_parse_unavailable'
            continue
        quantity = binding.get('quantity')
        if quantity == 'record_count':
            skip = binding.get('header_records', 0)
            if (type(skip) is not int or not 0 <= skip <= profile['records']
                    or (skip and not str(binding.get('header_basis') or '').strip())):
                check['reason'] = 'explicit_header_count_and_basis_required'
                continue
            expected_value = profile['records'] - skip
            check['unverified_assumptions'] = ['header_records_are_appropriate',
                                             'records_are_the_intended_unit']
        elif quantity == 'field_count':
            number = binding.get('record_number', 1)
            if type(number) is not int or not 1 <= number <= len(profile['first_records']):
                check['reason'] = 'record_number_outside_profile'
                continue
            expected_value = profile['first_records'][number - 1]['fields']
            check['unverified_assumptions'] = ['fields_are_the_intended_columns']
        else:
            check['reason'] = 'quantity_requires_another_validator'
            continue
        observed = metrics.get(metric)
        if type(observed) not in (int, float) or not math.isfinite(observed):
            check['reason'] = 'numeric_metric_missing'
            continue
        check.update(expected=expected_value, observed=observed,
                     status='passed' if observed == expected_value else 'failed',
                     scope='declared_table_quantity_only')
    checked = ({r['metric'] for r in checks if r.get('status') == 'passed'}
               - {r['metric'] for r in checks if r.get('metric') and r['status'] != 'passed'})
    failed = any(r['status'] == 'failed' for r in checks)
    return {'version': VERSION, 'checker_digest': file_digest(Path(__file__)),
        'assets_digest': fingerprint(assets), 'profiles': profiles, 'checks': checks,
        'status': 'failed' if failed else 'partial' if checked else 'unverified',
        'unverified_metrics': sorted(set(metrics) - checked),
        'seconds': time.monotonic() - started,
        'scientific_support': 'not_assessed', 'mechanism_support': 'not_assessed',
        'limits': ['no_header_or_unit_semantics_verified', 'no_data_integrity_claim',
                   'no_performance_or_mechanism_confirmation']}
