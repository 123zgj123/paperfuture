"""Pre-registered multi-setting studies: coverage that is not sample size.

One registered comparison on one bound world is one reading. The runtime could
express nothing else, so a contract asking for five was unrunnable — the fixture
check said ``insufficient_independent_samples`` and the frontier silently
offered no action at all. There was no way to say "run this comparison at each
of these five settings", and no way to combine the readings afterwards.

A study says exactly that, and freezes it before anything runs: which settings,
what a unit is, how the readings combine, when to stop, what happens when a
setting is missing, and what the conclusion may cover. Every field is inside the
registration digest, so changing any of them — a setting, a weight, the stopping
rule — is a different study with a different identity, not a better version of
this one.

Three counts are kept apart on purpose, because collapsing them is how a study
inflates itself:

``coverage_count``
    Registered settings with an admitted result. Deterministic coverage of a
    finite declared set; not a sample from any population.
``independent_unit_count``
    Distinct attested worlds behind those results. Different parameters, seeds,
    file names or world *names* do not make this number go up.
``repeat_count``
    Results beyond the first for a setting already covered. A repeat shows the
    measurement reproduces. It is not another unit.

Aggregation reads admitted sub-evidence and nothing else. It is recomputed from
those rows every time, so revoking one member changes the aggregate rather than
leaving a stale summary behind, and a missing setting leaves the study
incomplete rather than quietly narrowing what it claims.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

STUDY_VERSION = "multi-setting-study-v1"

#: Aggregation rules this version implements. ``mean`` states an average over
#: the registered settings; ``every_setting`` states a universal over them and
#: is not satisfied by an average that happens to clear the margin.
AGGREGATIONS = ("mean", "every_setting")
WEIGHTS = ("equal",)
STOPPING_RULES = ("all_registered_settings_once",)
MISSING_POLICIES = ("incomplete_until_every_setting_has_an_admitted_result",)

#: A finite declared set. Large enough to be a study, small enough that running
#: every setting stays a bounded, affordable obligation.
MIN_SETTINGS, MAX_SETTINGS = 2, 16

STUDY_FIELDS = frozenset({"version", "settings", "unit_definition", "aggregation",
                          "stopping_rule", "missing_result_policy", "coverage_scope"})
SETTING_FIELDS = frozenset({"id", "parameters"})


def study_of(execution: Mapping[str, Any] | None) -> dict[str, Any] | None:
    study = dict(execution or {}).get("study")
    return dict(study) if isinstance(study, Mapping) else None


def settings_digest(study: Mapping[str, Any] | None) -> str:
    """Identity of the registered setting list, short enough to sit in a claim."""
    rows = list((study or {}).get("settings") or ())
    payload = [{"id": str(row.get("id") or ""), "parameters": dict(row.get("parameters") or {})}
               for row in rows if isinstance(row, Mapping)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def study_errors(execution: Mapping[str, Any] | None,
                 parameter_errors: Any = None) -> list[str]:
    """Every way a study registration is not a study registration.

    ``parameter_errors`` is the operator's own parameter check, applied to each
    setting's merged parameters. A study cannot register a setting the operator
    could not have run on its own.
    """
    study = study_of(execution)
    if study is None:
        return []
    errors: list[str] = []
    if set(study) != STUDY_FIELDS:
        errors.append("study_registration_fields_incomplete_or_unknown")
        return errors
    if study.get("version") != STUDY_VERSION:
        errors.append("unsupported_study_version")
    rows = study.get("settings")
    if not isinstance(rows, list) or not MIN_SETTINGS <= len(rows) <= MAX_SETTINGS:
        errors.append("invalid_setting_count")
        rows = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != SETTING_FIELDS:
            errors.append("invalid_setting_structure")
            continue
        sid = row.get("id")
        if (not isinstance(sid, str) or not sid.strip() or not sid.isidentifier()):
            errors.append("invalid_setting_id")
            continue
        if sid in seen:
            errors.append("duplicate_setting_id")
        seen.add(sid)
        if not isinstance(row.get("parameters"), Mapping) or not row["parameters"]:
            errors.append("setting_without_parameters")
    aggregation = study.get("aggregation")
    if (not isinstance(aggregation, Mapping) or set(aggregation) != {"method", "weights"}
            or aggregation.get("method") not in AGGREGATIONS
            or aggregation.get("weights") not in WEIGHTS):
        errors.append("invalid_study_aggregation")
    if study.get("stopping_rule") not in STOPPING_RULES:
        errors.append("invalid_study_stopping_rule")
    if study.get("missing_result_policy") not in MISSING_POLICIES:
        errors.append("invalid_missing_result_policy")
    for name in ("unit_definition", "coverage_scope"):
        if not isinstance(study.get(name), str) or not study[name].strip():
            errors.append(f"missing_{name}")
    if callable(parameter_errors) and not errors:
        for row in rows:
            for problem in parameter_errors(setting_parameters(execution, str(row["id"]))):
                errors.append(f"setting_{row['id']}:{problem}")
    return errors


def setting_ids(execution: Mapping[str, Any] | None) -> list[str]:
    study = study_of(execution)
    return [str(row.get("id") or "") for row in (study or {}).get("settings") or ()
            if isinstance(row, Mapping)]


def setting_parameters(execution: Mapping[str, Any] | None, setting_id: str) -> dict[str, Any]:
    """The parameters one setting runs with: the base, overlaid by the setting.

    Overlaying rather than replacing keeps everything the study did not vary
    frozen at the registered value, so a setting cannot quietly change the
    intervention while claiming to change only the seed.
    """
    base = dict(dict(execution or {}).get("parameters") or {})
    study = study_of(execution)
    for row in (study or {}).get("settings") or ():
        if isinstance(row, Mapping) and str(row.get("id") or "") == str(setting_id):
            base.update(dict(row.get("parameters") or {}))
            return base
    return base


def study_rows(rows: Iterable[Mapping[str, Any]], *, claim_id: str = "",
               registration_digest: str = "", study_stage: str = "confirmatory",
               ) -> list[dict[str, Any]]:
    """Admitted sub-results that belong to exactly this registered study."""
    from ..validity import evidence_eligible

    out = []
    for row in rows or ():
        if (row.get("epistemic") != "WORLD" or row.get("invalidated")
                or not evidence_eligible(row)
                or str(row.get("study_stage") or "") != study_stage
                or not str(row.get("study_setting_id") or "")):
            continue
        if claim_id and str(row.get("discovery_idea_id") or "") != claim_id:
            continue
        if registration_digest and str(row.get("registration_digest") or "") != registration_digest:
            continue
        out.append(dict(row))
    return out


def coverage(rows: Iterable[Mapping[str, Any]], execution: Mapping[str, Any] | None,
             **selector: Any) -> dict[str, Any]:
    """Which registered settings have an admitted result, and what that is not."""
    registered = setting_ids(execution)
    members = study_rows(rows, **selector)
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in members:
        sid = str(row.get("study_setting_id") or "")
        if sid in registered:
            by_setting.setdefault(sid, []).append(row)
    covered = [sid for sid in registered if by_setting.get(sid)]
    worlds = {str(row.get("world_digest") or row.get("world_id") or "")
              for group in by_setting.values() for row in group}
    return {
        "settings_registered": list(registered),
        "settings_covered": covered,
        "missing_settings": [sid for sid in registered if sid not in covered],
        "coverage_count": len(covered),
        "independent_unit_count": len({item for item in worlds if item}),
        "repeat_count": sum(max(0, len(group) - 1) for group in by_setting.values()),
        "complete": bool(registered) and len(covered) == len(registered),
        "basis": ("deterministic coverage of a finite registered set; distinct "
                  "settings, seeds or file names are not independent units"),
    }


def aggregate(rows: Iterable[Mapping[str, Any]], execution: Mapping[str, Any] | None,
              **selector: Any) -> dict[str, Any]:
    """Combine admitted sub-results under the rule frozen at registration.

    The readings come from the rows' own arms, so an aggregate can only be as
    valid as its members and disappears with them. Nothing here reads a number a
    worker reported as an aggregate.
    """
    from ..validity import effect_verdict, separation

    study = study_of(execution)
    if study is None:
        return {}
    cover = coverage(rows, execution, **selector)
    members = {row["study_setting_id"]: row for row in
               sorted(study_rows(rows, **selector), key=lambda r: str(r.get("evidence_id") or ""))
               if str(row.get("study_setting_id") or "") in cover["settings_registered"]}
    margin = dict(execution or {}).get("margin")
    direction = str(dict(execution or {}).get("expected_direction") or "")
    per_setting = []
    for sid in cover["settings_registered"]:
        row = members.get(sid)
        if row is None:
            per_setting.append({"setting_id": sid, "status": "missing"})
            continue
        value = separation(row.get("treatment"), row.get("control"))
        per_setting.append({
            "setting_id": sid, "status": "measured",
            "evidence_id": str(row.get("evidence_id") or ""),
            "separation": value,
            "verdict": effect_verdict(value, margin, direction),
        })
    measured = [item for item in per_setting if item["status"] == "measured"]
    method = str((study.get("aggregation") or {}).get("method") or "")
    values = [item["separation"] for item in measured
              if isinstance(item["separation"], (int, float))]
    result = {
        "study_version": STUDY_VERSION,
        "settings_digest": settings_digest(study),
        "method": method,
        "weights": str((study.get("aggregation") or {}).get("weights") or ""),
        "coverage": cover,
        "per_setting": per_setting,
        "scope": str(study.get("coverage_scope") or ""),
        "evidence_ids": [item["evidence_id"] for item in measured],
    }
    if not cover["complete"]:
        # The missing-result policy is registered, and it is the only one this
        # version offers: an incomplete study concludes nothing.
        result.update(status="incomplete", verdict="",
                      reason=str(study.get("missing_result_policy") or ""))
        return result
    if method == "every_setting":
        verdicts = {item["verdict"] for item in measured}
        result.update(
            status="complete",
            verdict=("supports" if verdicts == {"supports"} else
                     "weakens" if verdicts == {"weakens"} else "uninformative"),
            reason=("a universal over the registered settings is carried by every "
                    "setting, never by their average"))
        return result
    mean = sum(values) / len(values) if values and len(values) == len(measured) else None
    result.update(
        status="complete" if mean is not None else "incomplete",
        value=mean,
        verdict=effect_verdict(mean, margin, direction) if mean is not None else "",
        reason=("an average over the registered settings; it does not establish "
                "any individual setting, nor any setting that was not registered"))
    return result
