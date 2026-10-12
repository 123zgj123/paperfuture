"""Resource accounting the scheduler can actually read.

Before this module the scheduler asked ``metrics["spent_cost"]`` against
``metrics["budget_cap"]`` — and nothing in the runtime ever wrote either field.
The budget filter was dead code, the token ledger lived inside the LLM client
where no research decision could see it, and wall-clock was recorded per action
and then dropped. An agent that cannot see its own remaining budget cannot
choose between "run the decisive experiment" and "open a new branch".

Everything here is derived by replay from ``measured_cost`` already stamped on
admitted events, so resumed sessions and rebuilt states agree.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


METERS = ("api_calls", "tokens", "wall_seconds", "probe_runs", "ticks")

DEFAULT_LIMITS: dict[str, float] = {}


def local_model_budget_meter(error: Exception) -> str:
    """TokenLedger's typed refusal, independent of provider/error prose."""
    from ..llm import LLMUnavailable
    if not isinstance(error, LLMUnavailable):
        return ''
    return {'llm_budget_api_calls': 'api_calls', 'llm_budget_token_cost': 'tokens'}.get(
        error.record.missing_capability, '')


def _limits(contract_limits: Mapping[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in dict(contract_limits or {}).items():
        if name in METERS and isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(name)] = float(value)
    return out


def spend_from_events(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Sum the costs that were actually measured, never estimated."""
    spent = {name: 0.0 for name in METERS}
    for row in rows or []:
        cost = row.get("measured_cost")
        if not isinstance(cost, Mapping):
            continue
        spent["tokens"] += float(cost.get("tokens") or 0)
        spent["wall_seconds"] += float(cost.get("wall_seconds") or 0)
        spent["api_calls"] += float(cost.get("api_calls") or 0)
    return spent


def services_exhausted(artifacts: Iterable[Mapping[str, Any]]) -> list[str]:
    """Services whose balance was reported spent in the current session.

    A spent balance does not come back within a session, so the run should end
    rather than spend its ticks rediscovering it. A rate limit or a dropped
    connection is weather: it is not counted here, and repeating one is already
    refused after two identical outcomes. Each session start is recorded, and a
    new session — typically after the account was topped up — tries again.
    """
    current: set[str] = set()
    for row in artifacts or []:
        kind = str(row.get("kind") or "")
        if kind == "research_service_session":
            current = set()
        elif kind == "action_attempt" and row.get("quota_exhausted") and row.get("service_unavailable"):
            current.add(str(row["service_unavailable"]))
    return sorted(current)


def compute_resources(state: Any) -> dict[str, Any]:
    """Canonical resource view: limits, measured spend, remaining, pressure."""
    from .contract import contract_from_state

    contract = contract_from_state(state)
    lifetime_limits = _limits(contract.resource_limits)
    sessions = [r for r in getattr(state, "artifacts", []) or []
                if r.get("kind") == "research_service_session" and "resource_limits" in r]
    session = sessions[-1] if sessions else {}
    session_limits = _limits(session.get("resource_limits"))
    started = dict(session.get("spent_at_start") or {})
    limits = dict(lifetime_limits)
    for name, cap in session_limits.items():
        ceiling = float(started.get(name) or 0) + cap
        limits[name] = min(limits.get(name, ceiling), ceiling)
    attempts = [
        dict(row)
        for row in getattr(state, "artifacts", []) or []
        if row.get("kind") == "action_attempt"
    ]
    evidence = [dict(row) for row in getattr(state, "evidence_records", []) or []]
    spent = spend_from_events(attempts)
    # Live runs stamp the ledger's real call count on every attempt. The
    # attempt count is only a floor for replayed or scripted states that
    # measured nothing, and it undercounts any action that makes several calls.
    measured_calls = any("api_calls" in (row.get("measured_cost") or {}) for row in attempts)
    spent["api_calls"] += float(sum(
        "api_calls" not in (row.get("measured_cost") or {}) for row in attempts))
    spent["probe_runs"] = float(
        sum(1 for row in attempts if row.get("action_type") in {"PROBE", "VERIFY"})
    )
    spent["ticks"] = float(int(getattr(state, "ticks", 0) or 0))

    remaining: dict[str, float] = {}
    fractions: dict[str, float] = {}
    for name, cap in limits.items():
        used = float(spent.get(name) or 0)
        remaining[name] = max(0.0, cap - used)
        ratios = []
        if name in lifetime_limits:
            total_cap = lifetime_limits[name]
            ratios.append(min(1.0, used / total_cap) if total_cap > 0 else 1.0)
        if name in session_limits:
            session_cap = session_limits[name]
            session_used = max(0.0, used - float(started.get(name) or 0))
            ratios.append(min(1.0, session_used / session_cap) if session_cap > 0 else 1.0)
        fractions[name] = max(ratios, default=0.0)
    pressure = max(fractions.values()) if fractions else 0.0
    exhausted = sorted(name for name, value in fractions.items() if value >= 1.0)
    # A connected client may own a smaller envelope than the runtime's cap.
    # Record its typed refusal without inventing spent calls or changing the
    # immutable contract. A new invocation gets a fresh client/session envelope.
    client_denied: set[str] = set()
    for row in getattr(state, 'artifacts', []) or []:
        if row.get('kind') == 'research_service_session':
            client_denied.clear()
        elif (row.get('kind') == 'action_attempt' and row.get('status') == 'budget_exhausted'
              and row.get('budget_meter') in {'api_calls', 'tokens'}):
            client_denied.add(row['budget_meter'])
    for meter in client_denied:
        remaining[meter] = 0.0
        fractions[meter] = 1.0
    exhausted = sorted(set(exhausted) | client_denied)
    pressure = max(fractions.values()) if fractions else 0.0
    stage_spent = _stage_spend(attempts, evidence)
    reserved = _reserved(state, remaining, stage_spent)
    return {
        "service_exhausted": services_exhausted(getattr(state, "artifacts", []) or []),
        "client_budget_exhausted": sorted(client_denied),
        "rule": "measured-spend-v1",
        "limits": limits,
        "lifetime_limits": lifetime_limits,
        "session_limits": session_limits,
        "session_spent": {name: max(0.0, round(value - float(started.get(name) or 0), 6))
                          for name, value in spent.items()},
        "spent": {name: round(float(spent.get(name) or 0), 6) for name in METERS},
        "remaining": {name: round(value, 6) for name, value in sorted(remaining.items())},
        "fraction_used": {name: round(value, 6) for name, value in sorted(fractions.items())},
        "budget_pressure": round(pressure, 6),
        "exhausted": exhausted,
        "basis": "measured_cost_from_admitted_events_not_estimates",
        "api_calls_basis": ("measured_model_calls" if measured_calls
                            else "action_attempt_count_floor"),
        "stage_spent": stage_spent,
        "reserved": reserved,
    }


def budget_state(state: Any) -> dict[str, Any]:
    raw = getattr(state, "resources", None)
    if isinstance(raw, Mapping) and raw.get("rule"):
        return dict(raw)
    return compute_resources(state)


def affordable(state: Any, *, cost: Mapping[str, float] | float = 0.0,
               action_type: str = "") -> bool:
    """Would this action's estimated cost cross a hard contract limit?

    Estimated cost is a spec constant, not a measurement. This only refuses
    actions that would exceed an explicitly declared limit.
    """
    budget = budget_state(state)
    limits = dict(budget.get("limits") or {})
    if budget.get("exhausted"):
        # Exhausting a model meter cannot prohibit a registered zero-call
        # local experiment. Time/probe/tick caps still stop all work.
        if not (isinstance(cost, Mapping)
                and set(budget['exhausted']) <= {'api_calls', 'tokens'}
                and all(float(cost.get(name, 0)) == 0 for name in budget['exhausted'])):
            return False
    if not limits:
        return True
    reserved = float((budget.get("reserved") or {}).get("api_calls") or 0)
    if action_type in DISCOVERY_ACTIONS and reserved:
        leftover = float((budget.get("remaining") or {}).get("api_calls") or 0)
        estimate = float(cost if not isinstance(cost, Mapping) else cost.get("api_calls") or 0)
        if leftover - estimate < reserved:
            return False
    if isinstance(cost, Mapping):
        for name, value in cost.items():
            cap = limits.get(str(name))
            if cap is None:
                continue
            if float(budget["spent"].get(str(name)) or 0) + float(value) > cap:
                return False
        return True
    cap = limits.get("api_calls")
    if cap is None:
        return True
    return float(budget["spent"].get("api_calls") or 0) + float(cost) <= cap


DISCOVERY_ACTIONS = frozenset({"IDEATE", "CRITIQUE", "EVOLVE_POLICY"})
COMMITTED_STAGES = frozenset({"confirmatory", "replication"})


def _stage_of(row: Mapping[str, Any]) -> str:
    kind = str(row.get("action_type") or "")
    stage = str(row.get("study_stage") or "")
    if stage in {"pilot", "calibration", "scientific_pilot", "confirmatory", "replication"}:
        return stage
    if kind in DISCOVERY_ACTIONS:
        return "discovery"
    if kind in {"SURVEY", "REVIEW"}:
        return "validation"
    if kind in {"PROBE", "EXPLORE"}:
        return "pilot"
    if kind == "VERIFY":
        return "replication" if row.get("replication_of") or row.get("replication_world_id") else "confirmatory"
    return "other"


def _stage_spend(attempts: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, float]:
    spent = {"discovery": 0.0, "validation": 0.0, "pilot": 0.0,
             "calibration": 0.0, "scientific_pilot": 0.0,
             "confirmatory": 0.0, "replication": 0.0, "other": 0.0}
    for row in attempts:
        cost = row.get("measured_cost") if isinstance(row.get("measured_cost"), Mapping) else {}
        spent[_stage_of(row)] += float(cost.get("api_calls", 1))
    return {name: round(value, 6) for name, value in spent.items()}


def _reserved(state: Any, remaining: Mapping[str, float], stage_spent: Mapping[str, float]) -> dict[str, float]:
    """Hold budget for confirmatory/replication once a lineage commits."""
    evidence = list(getattr(state, "evidence_records", []) or [])
    retired = {t.get('id') for t in getattr(state, 'theories', []) if t.get('status') == 'retired'}
    evidence = [r for r in evidence if not r.get('invalidated') and r.get('theory_id') not in retired]
    # Registered local operators do not spend model calls to confirm or replay.
    evidence = [r for r in evidence if (r.get('registration') or {}).get('version') not in {'computational-protocol-v2', 'computational-protocol-v3'}]
    committed = any(str(row.get("study_stage") or "") == "confirmatory" for row in evidence) or any(
        str(row.get("pilot_verdict") or "") == "PILOT_POSITIVE" for row in evidence)
    if not committed:
        return {"api_calls": 0.0, "reason": "no_committed_confirmatory_lineage"}
    leftover = float(remaining.get("api_calls") or 0)
    hold = min(leftover, max(2.0, leftover * 0.25))
    return {"api_calls": round(hold, 6), "reason": "reserved_for_confirmatory_and_replication"}


#: What fraction of the run's model budget each stage may spend. A stage that
#: has spent its share stops being offered *while something else can still
#: run* — never to the point of emptying the menu, because a project with no
#: move left is a worse outcome than one that over-spent on critique.
#:
#: The numbers come from measured live behaviour, not from taste. One mission
#: spent 30 calls with 8 IDEATEs and 10 CRITIQUEs and one measurement; four
#: others spent twenty-six ticks each on REVIEW / REDESIGN / THEORIZE over a
#: single idea while the critique that would have opened a second one sat at
#: the bottom of the queue. Both are a stage monopolising the run.
STAGE_CAPS = {
    "discovery": 0.40,    # IDEATE / CRITIQUE — divergence
    "validation": 0.35,   # SURVEY / REVIEW — literature and judgment
}

#: The same idea in the other currency. A run has two budgets and the traps
#: spend different ones: the divergence loop cost 27 calls, and the convergence
#: loop cost nothing at all — six REVIEWs, three REDESIGNs, three SYNTHESIZEs
#: and two VERIFYs, every one of them free, twenty-six ticks of the horizon
#: gone. A cap counted in calls cannot see that, so stages are capped in ticks
#: too. Measurement is deliberately uncapped: it is what the run is for.
STAGE_TICK_CAPS = {
    "discovery": 0.40,
    "validation": 0.30,
    "synthesis": 0.30,    # SYNTHESIZE / REDESIGN / THEORIZE / REFLECT / ARCHIVE
}

SYNTHESIS_ACTIONS = frozenset({"SYNTHESIZE", "REDESIGN", "THEORIZE", "REFLECT", "ARCHIVE"})


def action_stage(action: Any) -> str:
    """Which budgeted stage this action would spend from."""
    kind = str(getattr(action, "action_type", "") or "")
    stage = str((getattr(action, "extra", {}) or {}).get("study_stage") or "")
    if stage in COMMITTED_STAGES:
        return stage
    if stage in {"pilot", "calibration", "scientific_pilot"}:
        return "pilot"
    if kind in DISCOVERY_ACTIONS:
        return "discovery"
    if kind in {"SURVEY", "REVIEW"}:
        return "validation"
    if kind == "PROBE":
        return "pilot"
    if kind in SYNTHESIS_ACTIONS:
        return "synthesis"
    return "other"


def _stage_ticks(state: Any) -> dict[str, float]:
    """Ticks each stage has already taken, counted from recorded attempts."""
    from .recovery import TRANSPORT_STATUSES

    spent: dict[str, float] = {}
    for row in getattr(state, "artifacts", []) or ():
        # An outage took a tick from the clock but not from the stage: the turn
        # never happened as research, and a later session must be able to take
        # it. Same exemption the stall bound makes, for the same reason.
        if (row.get("kind") != "action_attempt"
                or str(row.get("status") or "") in TRANSPORT_STATUSES):
            continue
        name = action_stage(type("A", (), {
            "action_type": row.get("action_type"),
            "extra": {"study_stage": row.get("study_stage")}})())
        spent[name] = spent.get(name, 0.0) + 1.0
    return spent


def stage_budget_state(state: Any) -> dict[str, Any]:
    """Each capped stage's share, what it has spent, and whether it is over."""
    budget = budget_state(state)
    limit = float((budget.get("limits") or {}).get("api_calls") or 0)
    spent = dict(budget.get("stage_spent") or {})
    caps = dict(STAGE_CAPS)
    from .contract import contract_from_state

    declared = (getattr(contract_from_state(state), "resource_limits", {}) or {})
    for name in list(caps):
        override = declared.get(f"{name}_share")
        if isinstance(override, (int, float)) and 0 < float(override) <= 1:
            caps[name] = float(override)
    horizon = float((budget.get("limits") or {}).get("ticks") or 0)
    tick_caps = dict(STAGE_TICK_CAPS)
    for name in list(tick_caps):
        override = declared.get(f"{name}_tick_share")
        if isinstance(override, (int, float)) and 0 < float(override) <= 1:
            tick_caps[name] = float(override)
    ticks = _stage_ticks(state)
    rows = {}
    for name in sorted(set(caps) | set(tick_caps)):
        share = caps.get(name)
        allowance = limit * share if limit and share else 0.0
        used = float(spent.get(name) or 0)
        tick_share = tick_caps.get(name)
        tick_allowance = horizon * tick_share if horizon and tick_share else 0.0
        tick_used = float(ticks.get(name) or 0)
        rows[name] = {
            "share": share, "allowance": round(allowance, 6), "spent": used,
            "tick_share": tick_share, "tick_allowance": round(tick_allowance, 6),
            "ticks_spent": tick_used,
            "over": bool((limit and share and used >= allowance)
                         or (horizon and tick_share and tick_used >= tick_allowance)),
        }
    active = (contract_from_state(state).research_mode != 'open_research'
              or any(k.endswith('_share') for k in declared))
    return {"limit": limit, "horizon": horizon, "stages": rows,
            "active": active,
            "basis": "a stage that has spent its share of calls or of ticks "
                     "yields to one that has not"}


def within_stage_budget(state: Any, actions: list[Any]) -> list[Any]:
    """Drop over-cap actions, but never the last thing the project could do.

    The cap is what keeps a divergence loop or a review loop from eating a run.
    Emptying the menu would turn that into a blocked project, so an over-cap
    stage is only dropped while some other action remains runnable.
    """
    rows = stage_budget_state(state)
    over = {name for name, row in rows["stages"].items() if row["over"]}
    if not over:
        return list(actions)
    kept = [item for item in actions if action_stage(item) not in over]
    return kept or list(actions)


def compact_critique(remaining_calls: float | None) -> bool:
    """A cost policy, not a claim that joint and separate reviews are equivalent."""
    return (isinstance(remaining_calls, (int, float)) and not isinstance(remaining_calls, bool)
            and math.isfinite(remaining_calls) and remaining_calls <= 4)


def action_call_estimate(state: Any, action: Any) -> float:
    """Known call floors plus observed costs; not a hard token/cost guarantee."""
    from .scheduler import local_execution

    if local_execution(action, state) or action.action_type == 'EVOLVE_POLICY':
        return 0.0
    kind = action.action_type
    remaining = budget_state(state).get('remaining', {}).get('api_calls')
    if kind == 'CRITIQUE':
        return 1.0 if compact_critique(remaining) else 2.0
    if kind == 'IDEATE':
        return 1.0
    if kind == 'SURVEY' and action.extra.get('knowledge_request'):
        from .inquiry import inquiry_progress
        return float(inquiry_progress(state, action.extra['knowledge_request'])['estimated_model_calls'])
    floor = 2.0 if kind == 'SURVEY' else 1.0
    costs = [float(r['measured_cost']['api_calls']) for r in getattr(state, 'artifacts', [])
             if r.get('kind') == 'action_attempt' and r.get('action_type') == kind
             and isinstance((r.get('measured_cost') or {}).get('api_calls'), (int, float))
             and math.isfinite(r['measured_cost']['api_calls'])
             and r['measured_cost']['api_calls'] >= 0][-8:]
    return max(floor, sorted(costs)[math.ceil(.75 * len(costs)) - 1] if costs else floor)


def allocate_runnable(state: Any, actions: list[Any]) -> tuple[list[Any], str]:
    """Allocate the next decision from the runnable frontier, not lifetime shares.

    Explicit stage shares and metric-search contracts retain their declared
    policy. Open research defaults to executable dependency order and protects
    a next ideation/critique cycle from an unbounded opening survey. Missing
    scientific evidence does not disqualify exploratory candidates.
    """
    from .contract import contract_from_state

    contract = contract_from_state(state)
    if (contract.research_mode != 'open_research'
            or any(k.endswith('_share') for k in contract.resource_limits)):
        return within_stage_budget(state, actions), 'declared_stage_shares'
    remaining = budget_state(state).get('remaining', {}).get('api_calls')
    has_discovery = any(a.action_type in {'IDEATE', 'CRITIQUE'} for a in actions)
    from .exploratory import environment
    if environment(state) and any(a.action_type == 'IDEATE' for a in actions):
        # Inspect supplied assets before a generic opening literature tour.
        # Explicit question-driven requests remain eligible.
        actions = [a for a in actions if not (a.action_type == 'SURVEY'
            and not a.extra.get('knowledge_request')
            and not any(r.get('kind') == 'discovery_idea' for r in getattr(state, 'artifacts', [])))]
    if remaining is None or not has_discovery:
        return list(actions), 'rolling_dependencies'
    # A named hypothesis/objection can still justify targeted literature work.
    kept = [a for a in actions if not (
        a.action_type == 'SURVEY'
        and not any(a.extra.get(k) for k in ('theory_id', 'claim', 'objection_id'))
        and remaining - action_call_estimate(state, a) < 2)]
    return kept or list(actions), 'reserve_next_discovery_decision'


def allocation_view(state: Any, action: Any) -> dict[str, Any]:
    """Audit planned spend without inventing a scientific value score."""
    from .contract import contract_from_state

    contract = contract_from_state(state)
    return {'version': 'rolling-decision-budget-v2',
            'policy': ('declared_stage_shares' if contract.research_mode != 'open_research'
                       or any(k.endswith('_share') for k in contract.resource_limits)
                       else 'rolling_dependencies'),
            'estimated_model_calls': action_call_estimate(state, action),
            'remaining': dict(budget_state(state).get('remaining') or {}),
            'decision_package': dict(action.extra.get('knowledge_request') or {}),
            'cost_basis': 'known_call_floor_and_recent_costs_not_provider_guarantee',
            'value_basis': 'executable_dependencies_not_critic_score',
            'scientific_gain': 'unknown_until_external_measurement'}


def budget_phase(state: Any) -> str:
    """Coarse ordinal phase. Not a prediction of how much work remains.

    ``opening``  plenty of room: exploration and new branches are cheap.
    ``working``  normal.
    ``closing``  most of the budget is gone: prefer the decisive test over a
                 new lineage.
    ``exhausted`` nothing left to spend.
    """
    budget = budget_state(state)
    if not budget.get("limits"):
        return "unmetered"
    if budget.get("exhausted"):
        return "exhausted"
    pressure = float(budget.get("budget_pressure") or 0)
    if pressure >= 0.75:
        return "closing"
    if pressure >= 0.3:
        return "working"
    return "opening"
