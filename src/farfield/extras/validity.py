"""The evidence validity contract: what a recorded measurement is allowed to say.

A result used to be judged in five places — the probe worker, the belief
refresh, the quality projection, the structural reviewer and the readiness
gates — each reading the same row's flags and each deciding for itself what an
absent flag meant. ``world_consumed is not False`` read a placebo that never ran
as a pass; ``sample_size is None`` was a weak experiment to the quality
projection and a full-weight result to belief. A live run marked a hypothesis
supported on a separation that three of its five placebo worlds exceeded,
because the placebo comparison measured distance from the placebo *mean* and
forgot which side of it the result was on.

This module is the one definition, and it holds four rules.

1. **Undefined is not a value.** A ratio over nothing, a check that did not
   run, a number that is not finite, a fact that was never recorded: each is
   ``UNDEFINED``, and ``UNDEFINED`` never satisfies a check. No default passes.
2. **Judgments are derived from recorded facts.** A row records what was
   measured — the arms, the registered margin and direction, the placebo
   separations, the oracle placebo's arms, the units each arm measured. The
   verdict and every check are recomputed from those facts by the kernel on
   every replay, so a worker cannot assert one and an old log is re-judged by
   the rules in force.
3. **Direction is part of the claim.** An effect, a placebo comparison and a
   confidence interval are each read on a side, never as a distance.
4. **A worker can only weaken.** A failure recorded before this contract
   existed stands; a pass recorded then is a statement, not a fact, and is
   not believed.

Nothing here reads a model's output or calls one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

RULE = "evidence-validity-v1"

PASS = "pass"
FAIL = "fail"
UNDEFINED = "undefined"
NOT_APPLICABLE = "not_applicable"
BLOCKING = frozenset({FAIL, UNDEFINED})

SUPPORTS = "supports"
WEAKENS = "weakens"
UNINFORMATIVE = "uninformative"
DIRECTIONAL = frozenset({SUPPORTS, WEAKENS})

TREATMENT_HIGHER = "treatment_higher"
TREATMENT_LOWER = "treatment_lower"
DIRECTIONS = frozenset({TREATMENT_HIGHER, TREATMENT_LOWER})

# The probe worker destroys the world's structure this many times. Fewer
# recorded runs is not a smaller placebo distribution; it is an unfinished one.
MIN_PLACEBO_RUNS = 5

# Two recorded measurements closer than this are the same measurement.
EPS = 1e-9

# The checks, by what their failure means.
#
# A *measurement* check failing means the numbers do not say what they appear
# to — the registration is incomplete, execution was compromised, the arms met
# nothing, the measure is blind to its oracle, or the separation is one the
# script produces on a world with its structure destroyed. The verdict is then
# uninformative.
#
# An *attribution* check failing leaves the numbers standing but forbids
# crediting them to the claim: power unknown or too low, a competing mechanism
# left unablated, no registered claim to credit.
REGISTRATION = "registration"
EXECUTION = "execution"
OBJECT = "object"
ORACLE = "oracle"
PLACEBO = "placebo"
SAMPLE = "sample"
ABLATION = "ablation"
CLAIM = "claim"
MEASUREMENT_CHECKS = (REGISTRATION, EXECUTION, OBJECT, ORACLE, PLACEBO)
ATTRIBUTION_CHECKS = (SAMPLE, ABLATION, CLAIM)
CHECKS = MEASUREMENT_CHECKS + ATTRIBUTION_CHECKS

# What a worker wrote as a judgment on rows admitted before this contract. The
# kernel keeps these verbatim under ``worker_recorded`` so the record shows what
# was claimed, and reads them only where they can weaken a result.
WORKER_JUDGMENTS = (
    "verdict", "outcome", "separation", "world_consumed", "dv_blind",
    "cannot_corroborate", "claim_consistent", "mechanism_validated",
    "causal_scope", "scoped_verdict",
)


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

def number(value: Any) -> float | None:
    """A finite float, or None. A boolean is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator``; undefined over an empty denominator.

    Dividing by ``max(1, n)`` turned "no accepted updates" into a false
    acceptance fraction of 0.0, which a comparison then averaged in as a
    measurement.
    """
    return None if not denominator else float(numerator) / float(denominator)


def separation(treatment: Any, control: Any) -> float | None:
    """(treatment − control) relative to |control|; absolute when control is 0."""
    t, c = number(treatment), number(control)
    if t is None or c is None:
        return None
    base = abs(c)
    return (t - c) / base if base > 0 else t - c


def effect_verdict(sep: float | None, margin: Any, expected_direction: Any) -> str:
    """Supports, weakens or uninformative — a separation read on its side.

    Inside the margin the comparison failed to discriminate; that is a
    reading about the experiment, not the idea. An unregistered margin or
    direction cannot produce a verdict at all.
    """
    m = number(margin)
    if sep is None or m is None or m <= 0 or expected_direction not in DIRECTIONS:
        return UNINFORMATIVE
    if abs(sep) < m:
        return UNINFORMATIVE
    lower = sep < 0
    return SUPPORTS if lower == (expected_direction == TREATMENT_LOWER) else WEAKENS


def interval_verdict(
    ci_low: Any, ci_high: Any, margin: Any, expected_direction: Any
) -> tuple[str, str]:
    """(verdict, problem) for a confidence interval of separations.

    ``supports`` only when the whole interval clears the margin on the
    expected side, ``weakens`` when it clears it on the other. A reversed
    interval, a non-positive margin or an unregistered direction is not a
    narrow result — it is not a result, and the problem is named.
    """
    lo, hi, m = number(ci_low), number(ci_high), number(margin)
    if lo is None or hi is None:
        return UNINFORMATIVE, "the interval is not recorded as finite numbers"
    if lo > hi:
        return UNINFORMATIVE, "the interval's bounds are reversed"
    if m is None or m <= 0:
        return UNINFORMATIVE, "no positive pre-registered margin"
    if expected_direction not in DIRECTIONS:
        return UNINFORMATIVE, "no pre-registered direction"
    if expected_direction == TREATMENT_LOWER:
        supports, weakens = hi <= -m, lo >= m
    else:
        supports, weakens = lo >= m, hi <= -m
    if supports:
        return SUPPORTS, ""
    if weakens:
        return WEAKENS, ""
    return UNINFORMATIVE, (
        f"the interval [{lo:.6g}, {hi:.6g}] does not clear the margin {m:.6g} on one side")


def measure_is_blind(
    real: tuple[Any, Any], placebo: tuple[Any, Any], *, eps: float = EPS
) -> bool | None:
    """Did neither arm move when the oracle was scrambled? None if unmeasured."""
    values = [number(v) for v in (*real, *placebo)]
    if any(v is None for v in values):
        return None
    rt, rc, pt, pc = values
    return abs(rt - pt) <= eps and abs(rc - pc) <= eps


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    name: str
    status: str
    reason: str = ""

    @property
    def blocks(self) -> bool:
        return self.status in BLOCKING

    def to_dict(self) -> dict[str, str]:
        return {"check": self.name, "status": self.status, "reason": self.reason}


def placebo_check(real: Any, placebos: Iterable[Any] | None, margin: Any) -> Check:
    """Does the observed separation lie beyond every placebo world, by the margin?

    Each placebo run executes the registered script on a copy of the world
    with its structure destroyed, so the placebo separations are what the
    script produces without the object it claims to measure. A real effect
    has to lie outside that whole distribution *on its own side*: above the
    largest placebo separation when it is positive, below the smallest when
    it is negative, by at least the pre-registered margin.

    Beating all five is a floor, not a significance level — it is roughly a
    one-in-six permutation bound — and it is why five runs are required.
    """
    r, m = number(real), number(margin)
    runs = [number(value) for value in (placebos or ())]
    if r is None:
        return Check(PLACEBO, UNDEFINED, "the observed separation is not a finite number")
    if m is None or m <= 0:
        return Check(PLACEBO, UNDEFINED, "no positive pre-registered margin")
    if len(runs) < MIN_PLACEBO_RUNS or any(value is None for value in runs):
        return Check(PLACEBO, UNDEFINED,
                     f"{sum(v is not None for v in runs)} finite placebo runs recorded; "
                     f"{MIN_PLACEBO_RUNS} are required")
    finite = [float(v) for v in runs if v is not None]
    if r > 0:
        excess = r - max(finite)
    elif r < 0:
        excess = min(finite) - r
    else:
        return Check(PLACEBO, FAIL, "there is no separation to set against the placebo worlds")
    span = f"placebo separations span [{min(finite):.4g}, {max(finite):.4g}]"
    if excess >= m:
        return Check(PLACEBO, PASS, f"{r:.4g} lies beyond every placebo by {excess:.4g}; {span}")
    return Check(PLACEBO, FAIL,
                 f"{r:.4g} does not clear the placebo worlds by the margin {m:.4g}; {span}")


def sample_check(sample_size: Any, basis: Any = "", *, floor: float = 0.0) -> Check:
    """Were the units each arm measured reported, and are there enough?"""
    n = number(sample_size)
    if n is None:
        why = str(basis or "not_reported_by_the_registered_experiment")
        return Check(SAMPLE, UNDEFINED, f"the units each arm measured are unknown ({why})")
    if n < 1:
        return Check(SAMPLE, FAIL, "an arm measured no units")
    if floor and n < floor:
        return Check(SAMPLE, FAIL, f"n={n:g} per arm is below the contract's minimum of {floor:g}")
    return Check(SAMPLE, PASS, f"n={n:g} per arm")


def sample_floor(resource_limits: Mapping[str, Any] | None) -> float:
    """The contract's minimum informative sample, or 0 when it states none."""
    value = number(dict(resource_limits or {}).get("min_sample_size"))
    return value if value is not None and value > 0 else 0.0


def _recorded(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """What the worker wrote, frozen by the kernel at admission; else the row."""
    frozen = row.get("worker_recorded")
    return frozen if isinstance(frozen, Mapping) else row


def _registration(row: Mapping[str, Any]) -> Check:
    if row.get("exploratory"):
        return Check(REGISTRATION, FAIL, "post hoc analysis: nothing was registered before the arms ran")
    margin = number(row.get("margin"))
    if margin is None or margin <= 0:
        return Check(REGISTRATION, UNDEFINED, "no positive pre-registered margin is recorded")
    if row.get("expected_direction") not in DIRECTIONS:
        return Check(REGISTRATION, UNDEFINED, "no pre-registered direction is recorded")
    if separation(row.get("treatment"), row.get("control")) is None:
        return Check(REGISTRATION, UNDEFINED, "an arm is not a finite recorded number")
    return Check(REGISTRATION, PASS)


def _execution(row: Mapping[str, Any]) -> Check:
    if row.get("execution_status") not in (None, "ran"):
        return Check(EXECUTION, FAIL, "the registered experiment did not finish execution")
    if row.get("verify_count") and (row.get("reproduction_ok") is False
                                     or row.get("replay_ok") is False):
        return Check(EXECUTION, FAIL, "the recorded execution failed reproduction")
    gaps = [str(gap) for gap in row.get("validation_gaps") or [] if str(gap).strip()]
    if gaps:
        return Check(EXECUTION, FAIL, "; ".join(gaps))
    return Check(EXECUTION, PASS)


def _object(row: Mapping[str, Any]) -> Check:
    t, c = number(row.get("treatment")), number(row.get("control"))
    if t is None or c is None:
        return Check(OBJECT, NOT_APPLICABLE, "the arms are not recorded")
    if t == 0 and c == 0:
        return Check(OBJECT, FAIL, "both arms measured nothing; there is no comparison")
    return Check(OBJECT, PASS)


def _oracle(row: Mapping[str, Any]) -> Check:
    required = row.get("oracle_required")
    if required is None:
        said = _recorded(row).get("dv_blind")
        if said is True:
            return Check(ORACLE, FAIL, "recorded before this contract: the measure did not "
                                       "respond to the oracle placebo")
        if said is False:
            return Check(ORACLE, UNDEFINED, "recorded before this contract as responsive to the "
                                            "oracle, without the oracle placebo's arms")
        return Check(ORACLE, NOT_APPLICABLE, "no oracle placebo was required")
    if not required:
        return Check(ORACLE, NOT_APPLICABLE, "the claim's quantity is not oracle-relative")
    arms = row.get("oracle_placebo")
    arms = arms if isinstance(arms, Mapping) else {}
    blind = measure_is_blind((row.get("treatment"), row.get("control")),
                             (arms.get("treatment"), arms.get("control")))
    if blind is None:
        return Check(ORACLE, UNDEFINED, "the oracle placebo was required and its arms are not recorded")
    if blind:
        return Check(ORACLE, FAIL, "permuting the oracle moved neither arm; the measure does not "
                                   "read the quantity the claim names")
    return Check(ORACLE, PASS)


def _placebo(row: Mapping[str, Any], arithmetic: str) -> Check:
    if arithmetic not in DIRECTIONAL:
        return Check(PLACEBO, NOT_APPLICABLE, "no directional separation to set against placebo worlds")
    required = row.get("placebo_required")
    runs = row.get("placebo_separations")
    if required is False:
        return Check(PLACEBO, NOT_APPLICABLE, "the world has no structure a placebo could destroy")
    if required is None and runs is None:
        if _recorded(row).get("world_consumed") is False:
            return Check(PLACEBO, FAIL, "recorded before this contract: the effect was not "
                                        "separable from placebo worlds")
        return Check(PLACEBO, UNDEFINED, "no placebo comparison is recorded for a directional result")
    return placebo_check(separation(row.get("treatment"), row.get("control")),
                         runs if isinstance(runs, (list, tuple)) else (), row.get("margin"))


def _ablation(row: Mapping[str, Any]) -> Check:
    if row.get("ablation_required"):
        reason = str(row.get("ablation_reason") or "")
        return Check(ABLATION, FAIL, "a competing mechanism is not ablated"
                     + (f" ({reason})" if reason else ""))
    return Check(ABLATION, PASS)


def _claim(row: Mapping[str, Any], verdict: str, sep: float | None) -> tuple[Check, Any]:
    card = row.get("card") if isinstance(row.get("card"), Mapping) else {}
    spec = card.get("claim_spec")
    if row.get("claim_gate_error"):
        return Check(CLAIM, FAIL, f"the registered claim could not be applied: "
                                  f"{row.get('claim_gate_error')}"), None
    if not isinstance(spec, Mapping) or not spec:
        return Check(CLAIM, FAIL, "no claim specification was registered, so the result "
                                  "cannot be credited to a claim"), None
    from .claimspec import claim_spec_from_payload, project_evidence_to_claim

    diagnosis = row.get("diagnosis") if isinstance(row.get("diagnosis"), Mapping) else {}
    try:
        projection = project_evidence_to_claim(
            claim_spec_from_payload(spec),
            arithmetic_verdict=verdict,
            observed_effect={"treatment": row.get("treatment"), "control": row.get("control"),
                             "separation": None if sep is None else round(sep, 6)},
            metric=str(diagnosis.get("world_observable") or card.get("world_observable") or ""),
        )
    except (ValueError, TypeError, KeyError) as exc:
        return Check(CLAIM, FAIL, f"the registered claim cannot be read: {exc}"), None
    return Check(CLAIM, PASS), projection


# --------------------------------------------------------------------------
# The assessment of one probe row
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Assessment:
    """Everything a probe row is allowed to say, derived from what it recorded."""

    arithmetic: str
    verdict: str
    separation: float | None
    checks: tuple[Check, ...]
    mechanism_validated: bool = False
    projection: Any = None
    pilot_only: bool = False
    invalidated: bool = False

    def check(self, name: str) -> Check:
        return next(item for item in self.checks if item.name == name)

    @property
    def cannot_corroborate(self) -> bool:
        return self.invalidated or self.pilot_only or any(item.blocks for item in self.checks)

    @property
    def corroborates(self) -> bool:
        return self.verdict in DIRECTIONAL and not self.cannot_corroborate

    @property
    def measurement_valid(self) -> bool:
        return not self.invalidated and not any(self.check(name).blocks for name in MEASUREMENT_CHECKS)

    def dimension(self, names: tuple[str, ...]) -> str:
        """Conjunction of recorded checks, with absence distinct from failure."""
        statuses = {self.check(name).status for name in names}
        return FAIL if FAIL in statuses else UNDEFINED if UNDEFINED in statuses else PASS

    @property
    def correctness(self) -> str:
        """Execution integrity; running correctly does not validate the claim."""
        return FAIL if self.invalidated else self.dimension((EXECUTION,))

    @property
    def scientific_validity(self) -> str:
        return self.dimension((REGISTRATION, OBJECT, ORACLE, PLACEBO, ABLATION, CLAIM))

    @property
    def sufficiency(self) -> str:
        # These are explicit design requirements, not a statistical power model.
        return FAIL if self.pilot_only else self.dimension((SAMPLE,))

    def blocking(self) -> list[str]:
        return [f"{item.name}: {item.reason}" for item in self.checks if item.blocks]

    def fields(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """The derived columns of the row. Every one is overwritten on replay."""
        placebo, oracle = self.check(PLACEBO), self.check(ORACLE)
        out: dict[str, Any] = {
            "separation": None if self.separation is None else round(self.separation, 6),
            "verdict": self.verdict,
            "outcome": self.verdict,
            "world_consumed": {PASS: True, FAIL: False}.get(placebo.status),
            "dv_blind": {PASS: False, FAIL: True}.get(oracle.status),
            "object_absent": self.check(OBJECT).status == FAIL,
            "claim_consistent": self.check(CLAIM).status == PASS and self.measurement_valid,
            "cannot_corroborate": self.cannot_corroborate,
            "mechanism_validated": ("registered_discriminator_within_tested_conditions"
                                    if self.mechanism_validated else False),
            "causal_scope": ("within_tested_conditions" if self.mechanism_validated
                             else "unestablished"),
            "validity": {
                "rule": RULE,
                "arithmetic_verdict": self.arithmetic,
                "corroborates": self.corroborates,
                "measurement_valid": self.measurement_valid,
                "correctness": self.correctness,
                "scientific_validity": self.scientific_validity,
                "sufficiency": self.sufficiency,
                "checks": [item.to_dict() for item in self.checks],
                "blocking": self.blocking(),
            },
        }
        if self.mechanism_validated:
            out["mechanism_identified"] = True
        elif "mechanism_identified" in row:
            # What the registration established on its own: a discriminating
            # comparison in place of an ablation, or nothing.
            waived = row.get("discriminated_alternative") and not row.get("ablation_required")
            out["mechanism_identified"] = "registered_discriminating_comparison" if waived else False
        if self.projection is not None:
            record = self.projection.record
            out["scoped_verdict"] = record.verdict.value
            out["supports_scope"] = list(record.supports_scope)
            out["prohibited_inferences"] = list(record.prohibited_inferences)
        return out


def assess_probe(row: Mapping[str, Any], *, min_sample: float = 0.0) -> Assessment:
    """Judge one ProbeEvidence row from the facts it recorded.

    Total: every input the row may lack becomes an ``UNDEFINED`` check, never
    an exception, because the kernel runs this on replay and a malformed row
    must be refused rather than stop the log from being read.
    """
    registration = _registration(row)
    sep = separation(row.get("treatment"), row.get("control"))
    arithmetic = (effect_verdict(sep, row.get("margin"), row.get("expected_direction"))
                  if registration.status == PASS else UNINFORMATIVE)
    measurement = (registration, _execution(row), _object(row), _oracle(row),
                   _placebo(row, arithmetic))
    verdict = UNINFORMATIVE if any(item.blocks for item in measurement) else arithmetic
    claim, projection = _claim(row, verdict, sep)
    checks = measurement + (
        sample_check(row.get("sample_size"), row.get("sample_size_basis"), floor=min_sample),
        _ablation(row),
        claim,
    )
    pilot_only = row.get('study_stage') in {'pilot', 'calibration', 'scientific_pilot'}
    corroborates = not row.get("invalidated") and not pilot_only and verdict in DIRECTIONAL and not any(item.blocks for item in checks)
    validated = bool(row.get("discriminator_registered")) and verdict == SUPPORTS and corroborates
    return Assessment(arithmetic=arithmetic, verdict=verdict, separation=sep, checks=checks,
                      mechanism_validated=validated, projection=projection, pilot_only=pilot_only,
                      invalidated=bool(row.get("invalidated")))


def judge_probe_row(row: dict[str, Any], *, min_sample: float = 0.0) -> Assessment:
    """Freeze what the worker claimed, then overwrite every derived column.

    The first time the kernel sees a row it records the worker's own judgments
    under ``worker_recorded``; later passes read that frozen copy, so a column
    this function wrote is never mistaken for something the worker said.
    """
    if not isinstance(row.get("worker_recorded"), Mapping):
        row["worker_recorded"] = {key: row[key] for key in WORKER_JUDGMENTS if key in row}
    assessment = assess_probe(row, min_sample=min_sample)
    row.update(assessment.fields(row))
    return assessment


def check_status(row: Mapping[str, Any], name: str) -> str:
    """The status a derived row records for one check; undefined if never judged."""
    validity = row.get("validity") if isinstance(row.get("validity"), Mapping) else {}
    for item in validity.get("checks") or ():
        if isinstance(item, Mapping) and item.get("check") == name:
            return str(item.get("status") or UNDEFINED)
    return UNDEFINED


def check_reason(row: Mapping[str, Any], name: str) -> str:
    validity = row.get("validity") if isinstance(row.get("validity"), Mapping) else {}
    for item in validity.get("checks") or ():
        if isinstance(item, Mapping) and item.get("check") == name:
            return str(item.get("reason") or "")
    return ""


def evidence_eligible(row: Mapping[str, Any]) -> bool:
    """A retained observation may remain auditable after its rejection."""
    return row.get("epistemic") == "WORLD" and not row.get("invalidated")


def corroborates(row: Mapping[str, Any]) -> bool:
    """Did the kernel's derivation find this row able to move belief?"""
    validity = row.get("validity") if isinstance(row.get("validity"), Mapping) else {}
    return (evidence_eligible(row) and validity.get("rule") == RULE
            and validity.get("corroborates") is True)
