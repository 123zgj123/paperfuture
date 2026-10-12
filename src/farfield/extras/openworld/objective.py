"""The task's own score, and what counts as beating it.

Everything this runtime could previously say about a result was a per-hypothesis
judgment: the registered comparison supported, weakened, or did not separate.
That is enough to accept or reject one experiment and not enough to say *this is
the best we have on the root question*. Without a comparable task-level number
there is no notion of a breakthrough, no way to tell whether exploration is
getting better, and therefore no gradient for any policy that claims to improve
it. A system that cannot say what "better" means cannot improve.

The score is declared by the research contract, not invented by a worker:

    success_metric:
      name: between_species_separation
      source: separation          # which admitted evidence field carries it
      direction: higher_is_better
      min_improvement: 0.01       # below this, a change is not a breakthrough

Two rules keep it from becoming a way to launder progress:

1. **Only admitted WORLD evidence scores.** A GENERATED proposal, an
   unattested row, or a result the kernel refused to corroborate contributes
   nothing, however good its number looks.
2. **The metric reads a registered field.** It cannot be a fresh number a model
   supplied at scoring time, because then the scorer and the thing scored would
   be the same author.

A breakthrough here is a *measurement*: this admitted result is better than any
admitted result before it, by more than the declared epsilon. It is not a claim
of discovery, and it does not promote anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


HIGHER = "higher_is_better"
LOWER = "lower_is_better"
DIRECTIONS = (HIGHER, LOWER)

# Fields an admitted probe row may carry that a task metric is allowed to read.
# A metric naming anything else is refused rather than guessed at, so a contract
# cannot quietly point the score at a field a model chose to write.
SCORABLE_FIELDS = ("separation", "treatment", "control", "effect_size", "sample_size",
                   "directed_separation")

# A contrast that ran opposite to the pre-registered direction is not a strong
# result, it is a refutation. Scoring raw `separation` with higher-is-better let
# a `weakens` verdict at -0.18 stand as a live project's frontier: the metric
# could not tell "large effect as predicted" from "large effect the other way".
# `directed_separation` is the magnitude in the registered direction, floored at
# zero, so a reversal scores as no support rather than as the best result yet.
DIRECTED = "directed_separation"


@dataclass(frozen=True)
class SuccessMetric:
    """What "better" means for this project, declared before results exist."""

    name: str = ""
    source: str = ""
    direction: str = HIGHER
    min_improvement: float = 0.0

    @property
    def is_declared(self) -> bool:
        return bool(self.name and self.source in SCORABLE_FIELDS
                    and self.direction in DIRECTIONS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "source": self.source, "direction": self.direction,
            "min_improvement": float(self.min_improvement),
            "declared": self.is_declared,
            "basis": "reads_a_registered_field_of_admitted_world_evidence",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SuccessMetric":
        row = dict(payload or {})
        direction = str(row.get("direction") or HIGHER)
        try:
            epsilon = float(row.get("min_improvement") or 0.0)
        except (TypeError, ValueError):
            epsilon = 0.0
        return cls(
            name=str(row.get("name") or ""),
            source=str(row.get("source") or ""),
            direction=direction if direction in DIRECTIONS else HIGHER,
            min_improvement=max(0.0, epsilon),
        )

    def better(self, candidate: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        if self.direction == HIGHER:
            return candidate > incumbent + self.min_improvement
        return candidate < incumbent - self.min_improvement


def metric_of(state: Any) -> SuccessMetric:
    from .contract import contract_from_state

    return SuccessMetric.from_dict(contract_from_state(state).success_metric)


def _scorable(row: Mapping[str, Any]) -> bool:
    """Only a result the kernel actually admitted may carry a score."""
    return bool(
        str(row.get("role") or "") == "ProbeEvidence"
        and str(row.get("epistemic") or "") == "WORLD"
        and row.get("attested") and row.get("promotion_ok")
        and str(row.get("execution_status") or "") == "ran"
        and not row.get("invalidated")
        and not row.get("cannot_corroborate")
    )


def directed_separation(row: Mapping[str, Any]) -> float | None:
    """Separation in the pre-registered direction, floored at zero.

    The registration fixed which way the arms should differ before the probe
    ran. A result in that direction scores its magnitude; a result the other way
    scores zero, because it did not support the claim at all — it contradicted
    it, and a contradiction is not an improvement on the frontier.
    """
    value = row.get("separation")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    verdict = str(row.get("verdict") or row.get("outcome") or "")
    if verdict in {"weakens", "negative", "contradicted"}:
        return 0.0
    if verdict == "uninformative":
        return 0.0
    return max(0.0, float(value))


def score_of(metric: SuccessMetric, row: Mapping[str, Any]) -> float | None:
    if not metric.is_declared or not _scorable(row):
        return None
    if metric.source == DIRECTED:
        return directed_separation(row)
    value = row.get(metric.source)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def scored_results(state: Any) -> list[dict[str, Any]]:
    """Every admitted result that carries the task's declared score, in order."""
    metric = metric_of(state)
    out: list[dict[str, Any]] = []
    for row in getattr(state, "evidence_records", []) or []:
        value = score_of(metric, row)
        if value is None:
            continue
        out.append({
            "evidence_id": str(row.get("evidence_id") or ""),
            "theory_id": str(row.get("theory_id") or ""),
            "question_id": str(row.get("question_id") or ""),
            "world_id": str(row.get("world_id") or ""),
            "score": value,
            "admitted_at_tick": int(row.get("admitted_at_tick") or 0),
        })
    return out


def objective_state(state: Any) -> dict[str, Any]:
    """The project's standing against its own declared objective.

    ``frontier`` is the best admitted score so far. ``breakthroughs`` are the
    results that beat every result before them by more than the declared
    epsilon, which is how a project can tell that it is getting somewhere rather
    than merely staying busy.
    """
    metric = metric_of(state)
    if (getattr(state, 'contract', {}) or {}).get('research_mode', 'open_research') != 'metric_optimization':
        return {
            'rule': 'mission-scoped-objective-v2', 'metric': metric.to_dict(),
            'status': 'hypothesis_testing_not_metric_optimization',
            'frontier': None, 'results': 0, 'breakthroughs': [],
            'breakthrough_count': 0, 'ticks_since_breakthrough': None,
            'note': 'Metrics evaluate registered predictions. Their magnitude does not rank research questions.',
        }
    results = scored_results(state)
    if not metric.is_declared:
        return {
            "rule": "declared-task-objective-v1",
            "metric": metric.to_dict(),
            "status": "no_task_metric_declared",
            "note": ("Without a declared success metric this project can accept or "
                     "reject individual experiments but cannot say which result is "
                     "its best, and no policy over exploration has a gradient."),
            "frontier": None, "results": 0, "breakthroughs": [],
            "ticks_since_breakthrough": None,
        }
    frontier: float | None = None
    breakthroughs: list[dict[str, Any]] = []
    for row in results:
        if metric.better(row["score"], frontier):
            frontier = row["score"]
            breakthroughs.append({**row, "previous_frontier": None if not breakthroughs
                                  else breakthroughs[-1]["score"]})
    # Measurements that exist and did not qualify. Reporting only "results: 0"
    # hides that the project measured something and the value was refused, which
    # is the difference between "nothing happened" and "this did not hold up".
    unscored = []
    for row in getattr(state, "evidence_records", []) or []:
        if str(row.get("role") or "") != "ProbeEvidence" or _scorable(row):
            continue
        value = row.get(metric.source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            unscored.append({
                "evidence_id": str(row.get("evidence_id") or ""),
                "value": float(value),
                "refused_because": ("not_corroborating" if row.get("cannot_corroborate")
                                    else "not_admitted_world_evidence"),
            })
    last = breakthroughs[-1]["admitted_at_tick"] if breakthroughs else None
    ticks = int(getattr(state, "ticks", 0) or 0)
    return {
        "rule": "declared-task-objective-v1",
        "metric": metric.to_dict(),
        "status": "measured" if results else "declared_but_unmeasured",
        "frontier": frontier,
        "frontier_evidence_id": breakthroughs[-1]["evidence_id"] if breakthroughs else "",
        "results": len(results),
        "measured_but_unscored": unscored,
        "breakthroughs": breakthroughs,
        "breakthrough_count": len(breakthroughs),
        "ticks_since_breakthrough": None if last is None else max(0, ticks - last),
        "note": ("A breakthrough is a measurement that beat every admitted "
                 "measurement before it. It is not a claim of discovery and it "
                 "promotes nothing."),
    }


def is_breakthrough(before: Any, after: Any) -> bool:
    """Did this tick move the project's own frontier?"""
    left = objective_state(before).get("breakthrough_count") or 0
    right = objective_state(after).get("breakthrough_count") or 0
    return int(right) > int(left)
