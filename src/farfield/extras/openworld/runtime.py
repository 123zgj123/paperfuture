"""Production research entry: OpenWorld owns the next action."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping
from ...services.workspace import workspace_lease

from . import OpenWorld
from .contract import ResearchContract
from .events import make_event
from .executors import fixture_from_root
from .worldio import bind_fixture, is_attested
from .exploration import trajectory_metrics
from .journal import research_journal
from .progress import should_terminate
from .reducer import apply_events, rebuild_state, state_digest
from .termination import (
    BLOCKED,
    BUDGET_EXHAUSTED,
    JUDGED_BELOW_CONTRACT,
    MISSION_NOT_DECLARED,
    NO_VIABLE_PATH,
    RUN_SEGMENT_DONE,
    SERVICE_UNAVAILABLE,
    readiness,
    run_outcome,
    task_delivery,
)

#: Endings that stop the segment without settling the science. Each one
#: leaves work a later session could pick up, so each one writes a handoff.
HANDOFF_STATUSES = frozenset({BLOCKED, NO_VIABLE_PATH, BUDGET_EXHAUSTED,
                              SERVICE_UNAVAILABLE, MISSION_NOT_DECLARED,
                              JUDGED_BELOW_CONTRACT})


def _seed_world(workspace: Path, world_id: str = "W0", topic: str = "") -> Path:
    root = workspace / "worlds" / world_id
    if root.is_dir() and any(root.iterdir()):
        return root
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": world_id,
        "topic": topic,
        "provenance": "generated",
        "records": [],
    }
    (root / "world.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root


def default_contract(topic: str, **limits: Any) -> ResearchContract:
    """The contract a caller gets when they only supplied a topic string.

    A topic is not a research project. Even the default has to name what would
    count as success, what quality bar is being aimed at, and what the run is
    allowed to spend — otherwise there is nothing for the controller to be
    faithful to fifty actions later.
    """
    return ResearchContract.from_dict({
        "root_question": topic,
        "success_target": (
            "admitted WORLD evidence that answers the root question within its "
            "stated scope, surviving review"),
        "target_quality": "internal",
        "allowed_reframes": list(ResearchContract().allowed_reframes),
        "resource_limits": {k: v for k, v in limits.items() if v is not None},
        "termination_requirements": [
            "central_question_answered", "no_fatal_objections", "result_review_current",
            "evidence_scope_matches_claim", "no_unauthorized_goal_drift",
        ],
    })


def bootstrap_research(
    topic: str,
    workspace: Path,
    *,
    world: Any = None,
    world_id: str = "",
    contract: Any = None,
) -> OpenWorld:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    env = OpenWorld.load(workspace)
    if env.state.goal:
        if env.state.goal != topic:
            raise ValueError("workspace already belongs to a different research topic")
        if world is not None or world_id:
            requested = str(getattr(world, "id", "") or world_id or world)
            if requested not in {str(row.get("id")) for row in env.state.world_versions}:
                raise ValueError("workspace belongs to a different world lineage")
        # Snapshots are derived caches. Replaying adds new research views to
        # older runs without rewriting their immutable scientific event log.
        rebuilt = rebuild_state(env.log)
        if state_digest(env.state) != state_digest(rebuilt):
            env.state = rebuilt
            env.persist()
        return env
    env.state.goal = topic
    if contract is None:
        contract = default_contract(topic)
    elif not isinstance(contract, ResearchContract):
        contract = ResearchContract.from_dict(contract)
    if contract.root_question and contract.root_question != topic:
        raise ValueError("the research contract's root question must be the run topic")
    wid = str(world_id or getattr(world, "id", "") or "W0")
    if world is not None or world_id:
        fixture = bind_fixture(workspace, world, world_id)
        env.state.world_id = fixture.id
        env.state.world_versions.append(
            {
                "id": env.state.world_id,
                "world_id": env.state.world_id,
                "digest": fixture.digest,
                "observables": fixture.schema or "artifact_inventory",
                "path": str(fixture.root),
                # Whether an experiment can run on it. The evidence runtime's
                # registered hypotheses are compiled only for an attested world.
                "attested": bool(is_attested(fixture)),
            }
        )
    else:
        root = _seed_world(workspace, wid, topic=topic)
        fixture = fixture_from_root(root, wid)
        env.state.world_id = wid
        env.state.world_versions.append(
            {
                "id": wid,
                "world_id": wid,
                "digest": fixture.digest,
                "observables": "",
                "path": str(root),
                # A placeholder for a project with no world yet. Discovery runs
                # without one; no registered experiment can.
                "provenance": "generated",
                "attested": False,
            }
        )
    qid = "Q1"
    observable = fixture.schema or "artifact_inventory"
    # Only an actual snapshot artifact enables the specialized observer.
    if "validator_snapshots.json" in fixture.files:
        observable = "validator_snapshots"
    env.state.world_versions[-1]["observables"] = (
        observable if fixture.schema or "validator_snapshots.json" in fixture.files else ""
    )
    env.state.add_question(
        question_id=qid,
        text=topic,
        required_observable=observable,
        lineage_id="line-q1",
        epistemic="GENERATED",
    )
    env._ensure_baseline()
    env.state = apply_events(
        env.state,
        [
            make_event("GoalSet", {"goal": topic}),
            make_event("ContractSet", contract.to_dict()),
            make_event(
                "WorldCreated",
                dict(env.state.world_versions[-1]),
            ),
            make_event(
                "QuestionCreated",
                {
                    "id": qid,
                    "text": env.state.open_questions[0]["text"] if env.state.open_questions else topic,
                    "required_observable": observable,
                    "lineage_id": "line-q1",
                    "epistemic": "GENERATED",
                },
            ),
        ],
        log=env.log,
    )
    env.persist()
    return env


@workspace_lease
def run_research(
    topic: str,
    *,
    workspace: Path,
    horizon: int = 20,
    world: Any = None,
    world_id: str = "",
    stop_on_exhaust: bool = True,
    client: Any = None,
    feed: Any = None,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    api_calls: float | None = 100,
    mode: str = "live",
    heldout_worlds: tuple[Any, ...] = (),
    policy_log: Path | None = None,
    policy_file: Path | None = None,
    policy_evaluation: Any = None,
    policy_suite: dict[str, list[str]] | None = None,
    policy_evaluator: Any = None,
    contract: Any = None,
    corpus_id: str = "",
    remote_search: Any = None,
    mission_claim: Mapping[str, Any] | None = None,
    stage_shares: Mapping[str, float] | None = None,
    quality_requirements: Mapping[str, str] | None = None,
    panel_clients: Mapping[str, Any] | None = None,
    panel_models: Mapping[str, str] | None = None,
    knowledge_model: str = "",
    exploration_assets: Mapping[str, Path] | None = None,
    native_evaluator: Any = None,
    allowed_actions: tuple[str, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    """One scientific controller; all research services are existing workers."""
    from ..llm import LLMClient, LLMUnavailable, TokenLedger, resolve_backend
    from .workers import LegacyResearchWorkers

    call_limit = 100.0 if api_calls is None else float(api_calls)
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    if not math.isfinite(call_limit) or call_limit < 0:
        raise ValueError("api_calls must be finite and nonnegative")
    if contract is None:
        # Invocation limits belong to a segment; explicit contract limits
        # remain lifetime caps and are never replenished by resuming.
        # `stage_shares` sets how much of the run each stage may take, e.g.
        # {"discovery_share": 0.3, "synthesis_tick_share": 0.2}; `mission_claim`
        # declares the mission as a registered comparison. Both belong to the
        # contract, so they are fixed at the first segment like the rest of it.
        contract = default_contract(topic, **{
            str(name): float(value) for name, value in dict(stage_shares or {}).items()})
        if mission_claim:
            contract = replace(contract, mission_claim=dict(
                ResearchContract.from_dict({"mission_claim": dict(mission_claim)}).mission_claim))
        if quality_requirements:
            # The bar a caller sets on top of the default, e.g.
            # {"novelty": "partial", "importance": "partial"} to ask for a judged
            # idea. Without this a run could only ever be held to `internal`.
            contract = replace(contract, quality_requirements={
                str(name): str(level) for name, level in dict(quality_requirements).items()})
    env = bootstrap_research(topic, Path(workspace), world=world,
                             world_id=str(world_id or ""), contract=contract)
    if allowed_actions is not None:
        from .actions import SPECS
        if any(name not in SPECS for name in allowed_actions):
            raise ValueError('unknown_action_in_host_tool_policy')
        policy = {'kind': 'research_tool_policy', 'allowed_actions': sorted(set(allowed_actions)),
                  'authority': 'host_configuration', 'epistemic': 'GENERATED'}
        previous = [r for r in env.state.artifacts if r.get('kind') == 'research_tool_policy']
        if previous and previous[-1]['allowed_actions'] != policy['allowed_actions']:
            raise ValueError('host_tool_policy_changed_on_resume')
        if not previous:
            env.state = apply_events(env.state, [make_event('ArtifactAdded', policy)], log=env.log)
    from .exploratory import bind_assets, environment, ExplorationWorkers
    if exploration_assets is not None:
        bound = bind_assets(Path(workspace), exploration_assets)
        existing = environment(env.state)
        if existing and existing['assets_digest'] != bound['assets_digest']:
            raise ValueError('exploration_assets_changed_on_resume')
        if not existing:
            env.state = apply_events(env.state, [make_event('ArtifactAdded', bound)], log=env.log)
    if environment(env.state):
        env.connect_workers(ExplorationWorkers(Path(workspace)))
    if native_evaluator is not None:
        from ...services.native_evaluation import bind_evaluator
        bound = bind_evaluator(Path(workspace), native_evaluator)
        existing = [r for r in env.state.artifacts if r.get('kind') == 'native_evaluation_environment']
        if existing and existing[-1]['contract_digest'] != bound['contract_digest']:
            raise ValueError('native_evaluator_changed_on_resume')
        if not existing:
            env.state = apply_events(env.state, [make_event('ArtifactAdded', bound)], log=env.log)
    from .dependencies import integrity_retractions
    env.state = apply_events(env.state, integrity_retractions(Path(workspace), env.state), log=env.log)
    client_gap = ""
    if client is None and call_limit > 0 and horizon > 0:
        try:
            backend = resolve_backend(model=model, base_url=base_url, api_key=api_key)
            call_limit = 100.0 if api_calls is None else float(api_calls)
            ledger = TokenLedger(api_calls=call_limit, token_cost=call_limit * 16000)
            client = LLMClient(backend, Path(workspace) / "llm_cache", mode=mode, ledger=ledger)
            # A panel role on another model shares this run's endpoint, cache and
            # ledger: a second opinion is not a second budget.
            panel_clients = {**{
                str(role): LLMClient(replace(backend, model=str(name)),
                                     Path(workspace) / "llm_cache", mode=mode, ledger=ledger)
                for role, name in dict(panel_models or {}).items()
                if str(name) and str(name) != backend.model}, **dict(panel_clients or {})}
        except LLMUnavailable as exc:
            client_gap = str(exc)
    import os
    knowledge_name = str(knowledge_model or os.environ.get('FARFIELD_KNOWLEDGE_MODEL', '')).strip()
    knowledge_client = None
    if knowledge_name and isinstance(client, LLMClient):
        knowledge_client = LLMClient(replace(client.backend, model=knowledge_name),
            Path(workspace) / 'llm_cache', mode=client.mode, ledger=client.ledger)
    env.connect_workers(LegacyResearchWorkers(client=client, feed=feed, workspace=Path(workspace),
                                              knowledge_client=knowledge_client))
    if client is not None:
        # Discovery needs a model and nothing else — no literature, no world.
        from .ideation import DiscoveryWorkers

        env.connect_workers(DiscoveryWorkers(client=client, workspace=Path(workspace),
                                             corpus_id=str(corpus_id or ""),
                                             remote_search=remote_search,
                                             panel_clients=panel_clients))
    primary_world = dict(next(row for row in env.state.world_versions if row.get("id") == env.state.world_id))
    for requested in heldout_worlds:
        fixture = bind_fixture(Path(workspace), requested, "")
        env.state = apply_events(env.state, [make_event("WorldCreated", {
            "id": fixture.id, "world_id": fixture.id, "digest": fixture.digest,
            "observables": fixture.schema, "research_role": "heldout", "role": "heldout",
            "schema": fixture.schema, "attested": bool(is_attested(fixture)),
            "path": str(fixture.root)}), make_event("WorldCreated", primary_world)], log=env.log)
    from ..routing import load_policy
    policy = load_policy(Path(policy_file)) if policy_file is not None else None
    backend = getattr(client, "backend", None)
    env.state = apply_events(env.state, [make_event("ArtifactAdded", {
        "kind": "research_service_session", "id": f"session-{env.log.next_seq()}",
        "resource_limits": {"api_calls": call_limit, "ticks": float(horizon)},
        "spent_at_start": dict(env.state.resources.get("spent") or {}),
        "model": str(getattr(backend, "model", model)), "base_url": str(getattr(backend, "base_url", base_url)),
        "mode": str(getattr(client, "mode", mode))}), make_event("ArtifactAdded", {
        "kind": "research_policy_snapshot", "policy": policy,
        "activation": "frozen_for_this_mission_segment"})], log=env.log)
    ledger = getattr(client, "ledger", None)
    if isinstance(ledger, TokenLedger):
        ledger.api_calls = min(ledger.api_calls, ledger.spent_calls +
                              float(env.state.resources["remaining"].get("api_calls", call_limit)))
    env.persist()
    yield {
        "stage": "openworld_start",
        "controller": "OpenWorld",
        "goal": env.state.goal,
        "world_id": env.state.world_id,
        "harness_version": env.harness.version_id,
        "horizon": int(horizon),
        "output_kind": "scientific_runtime",
        "research_complete": False,
        "capability_note": client_gap or "Existing research workers connected; admission remains evidence-gated.",
    }
    horizon_reached, no_action = True, False
    service_failure: dict[str, Any] = {}
    # The segment's cap, counted from the ticks a resumed workspace already has.
    cap = int(env.state.ticks) + max(0, int(horizon))
    for _ in range(max(0, int(horizon))):
        delivery = task_delivery(env.state)
        if stop_on_exhaust and delivery['complete'] and delivery['stop_on_acceptance']:
            yield {'stage': 'terminate', 'reason': 'task_delivery_complete',
                   'task_complete': True, 'task_delivery': delivery,
                   'research_complete': readiness(env.state).ready, 'ticks': env.state.ticks}
            horizon_reached = False
            break
        verdict = readiness(env.state)
        if stop_on_exhaust and verdict.ready:
            yield {'stage': 'terminate', 'reason': 'research_ready', 'ticks': env.state.ticks,
                   'research_status': verdict.status, 'research_complete': True, 'failed_gates': []}
            horizon_reached = False
            break
        reason = should_terminate(env.state, horizon=cap) if stop_on_exhaust else None
        if reason and reason != "budget" and stop_on_exhaust:
            # A frontier reason stops the loop. It never settles the science, so
            # the readiness verdict is reported alongside it.
            verdict = readiness(env.state)
            yield {"stage": "terminate", "reason": reason, "ticks": env.state.ticks,
                   "research_status": verdict.status, "research_complete": verdict.ready,
                   "failed_gates": list(verdict.failed_gates)}
            horizon_reached = False
            break
        if env.state.resources.get("exhausted"):
            from .scheduler import admissible, local_execution
            local, _ = admissible(env.state, [a for a in env.eligible() if local_execution(a, env.state)])
            if not local:
                verdict = readiness(env.state)
                yield {"stage": "terminate", "reason": "budget_exhausted", "ticks": env.state.ticks,
                       "research_status": BUDGET_EXHAUSTED, "research_complete": False,
                       "failed_gates": list(verdict.failed_gates)}
                horizon_reached = False
                break
        event = env.tick()
        yield event
        result = event.get('result') or {}
        if result.get('status') == 'service_unavailable':
            # The client already exhausted its request-level retries. A new
            # scientific action must not retry the same outage or relabel it
            # as a research blockage. A later invocation may try again.
            service_failure = dict(result)
            horizon_reached = False
            verdict = readiness(env.state)
            yield {'stage': 'terminate', 'reason': 'service_unavailable',
                   'ticks': env.state.ticks, 'research_status': SERVICE_UNAVAILABLE,
                   'research_complete': False, 'failed_gates': list(verdict.failed_gates),
                   'service': result.get('service', ''), 'detail': result.get('reason', ''),
                   'quota_exhausted': bool(result.get('quota_exhausted'))}
            break
        if event.get("scheduled") is None:
            no_action = True
            verdict = readiness(env.state, no_executable_action=True)
            yield {"stage": "terminate", "reason": "no_executable_frontier_action",
                   "ticks": env.state.ticks, "research_status": verdict.status,
                   "research_complete": verdict.ready,
                   "failed_gates": list(verdict.failed_gates)}
            horizon_reached = False
            break
    if (policy_log is not None and policy_file is not None and not service_failure
            and not set(env.state.resources.get('exhausted') or []) & {'api_calls', 'tokens'}):
        from ..routing import append_mission, load_log, maybe_update, _validated_outcome, PolicyError
        import hashlib
        mission_id = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()
        cards = []
        for evidence in env.state.evidence_records:
            if evidence.get("role") != "ProbeEvidence":
                continue
            theory = next((t for t in env.state.theories if t.get("id") == evidence.get("theory_id")), {})
            row = dict(evidence, ledger="scientific", operator=theory.get("operator_reused") or "unknown")
            try:
                _validated_outcome(row)
            except (PolicyError, ValueError, TypeError):
                continue
            cards.append(row)
        # Budget exhaustion settles a run, not the scientific claims it explored.
        # Repeated resumes of the same mission never count as independent failures.
        if cards and not any(m.get("mission_id") == mission_id for m in load_log(Path(policy_log))):
            failures = [{"id": f"failure-{r['evidence_id']}", "class": "SCIENTIFIC_FAILURE",
                "signature": "non_discriminating_design" if r.get("verdict") == "uninformative" else "weakened_mechanism",
                "evidence_id": r["evidence_id"]} for r in cards if r.get("verdict") in {"weakens", "uninformative"}]
            append_mission(Path(policy_log), {"mission_id": mission_id, "status": "completed",
                "research_complete": False, "topic": topic, "cards": cards, "failures": failures,
                "scientific_state_digest": state_digest(env.state)})
        if policy_evaluation is None and policy_suite is not None:
            from ..routing import evaluate_research_policy
            policy_evaluation = evaluate_research_policy(Path(policy_log), Path(policy_file),
                suite=policy_suite, evaluator=policy_evaluator)
        decision = maybe_update(Path(policy_log), Path(policy_file), evaluation=policy_evaluation)
        yield {"stage": "research_policy", **decision}
    rebuilt = rebuild_state(env.log)
    verdict = readiness(env.state, no_executable_action=no_action)
    if service_failure:
        verdict = replace(verdict, status=SERVICE_UNAVAILABLE,
                          reasons=verdict.reasons + (str(service_failure.get('reason') or 'model service unavailable'),))
    # Two different questions, reported apart: is this worth pursuing, and has
    # it been shown. A blocked evidence path does not undo a discovery.
    from .discovery import discovery_summary
    from .ideation import write_deliverables

    discovery = discovery_summary(env.state)
    deliverables = write_deliverables(Path(workspace), env.state)
    status_text = (
        "# Research status / 研究状态\n\n"
        f"Topic: {env.state.goal}\n\n"
        "## Discovery / 发现\n\n"
        f"{discovery['headline']}\n\n"
        + (f"Discovery needs a research model, and none is connected: {client_gap}\n\n"
           if client is None else "")
        + "".join(f"{line}\n" for line in discovery["narrative"] + discovery["closing"])
        + f"\nDiscovery status: {discovery['discovery_status']} · ideas {discovery['ideas']} · "
        f"validated {discovery['validated']} · exploration policy {discovery['policy_version']}\n\n"
        f"Selected idea: {discovery['selected_idea'] or '—'} · research promise "
        f"{discovery['research_promise']} · evidence status {discovery['evidence_status']}\n\n"
        f"Deliverables: {sorted(deliverables)}\n\n"
        "## Evidence / 证据\n\n"
        f"Evidence readiness: {verdict.status} (ready={verdict.ready})\n\n"
        f"Unmet gates: {list(verdict.failed_gates)}\n\n"
        f"Quality gap: {list((env.state.quality or {}).get('quality_gap') or [])}\n\n"
        f"Open objections: {[r.get('review_id') for r in env.state.reviews if r.get('status') == 'open']}\n\n"
        f"Status: {verdict.status}; answer: {(env.state.quality.get('question_answer') or {}).get('status', 'unresolved')}. "
        "描述性观测不能确认原始假设；验证结论只适用于已测试条件。\n\n"
        "Next actions are selected from ScientificState; missing services or frozen "
        "worlds remain explicit blockers. Speculative hypotheses are not verified knowledge.\n\n"
        f"Open questions: {[row.get('text') for row in env.state.open_questions]}\n"
        f"Blocked questions: {[row.get('text') for row in env.state.blocked_questions]}\n"
        f"Missing capabilities: {env.state.missing_capabilities}\n"
        f"Speculative frontier: {env.state.speculative_frontier}\n"
        f"Verified findings: {list(env.state.verified_research_state)}\n"
    )
    gaps = sorted({str(row.get("reason")) for row in env.history if row.get("reason")})
    status_text += f"Execution gaps: {gaps}\n"
    observations = [row for row in env.state.evidence_records if row.get("role") == "QuestionEvidence"]
    if observations:
        status_text += "\n## Exploratory observations / 探索性观测\n\n"
        status_text += "```json\n" + json.dumps(observations, ensure_ascii=False, indent=2) + "\n```\n"
    (Path(workspace) / "RESEARCH_STATUS.md").write_text(status_text, encoding="utf-8")
    # The proposal has IDEA.md and the critic's view has DISCOVERY.md. This is
    # the third document, and the one the run exists to produce: what was
    # tested, what was read, and what that does and does not establish.
    from .journal import render_result

    (Path(workspace) / "RESULT.md").write_text(render_result(env.state), encoding="utf-8")
    outcome = run_outcome(env.state, horizon_reached=horizon_reached,
                          no_executable_action=no_action)
    # A segment that cannot continue leaves a package behind rather than a
    # verdict. It is a local file: nothing is sent, and no action type exists
    # to send it. A ready run has nothing to hand off.
    handoff: dict[str, Any] = {}
    if outcome["status"] in HANDOFF_STATUSES and not outcome["research_complete"]:
        from .recovery import handoff_package

        handoff = handoff_package(env.state, workspace=str(Path(workspace)),
                                  reason=outcome["status"].casefold())
        (Path(workspace) / "RESEARCH_HANDOFF.json").write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if service_failure:
        outcome = {**outcome, **verdict.to_dict(), 'research_complete': False,
                   'segment': RUN_SEGMENT_DONE, 'service_unavailable': {
                       'service': service_failure.get('service', ''),
                       'reason': service_failure.get('reason', ''),
                       'quota_exhausted': bool(service_failure.get('quota_exhausted'))}}
    yield {
        "stage": "done",
        "status": "ready" if outcome["research_complete"] else
                  "task_complete" if outcome['task_complete'] else "incomplete",
        "task_complete": outcome['task_complete'],
        "task_delivery": outcome['task_delivery'],
        "research_status": outcome["status"],
        "research_complete": outcome["research_complete"],
        "readiness": outcome,
        "segment": outcome.get("segment", RUN_SEGMENT_DONE),
        "decision_journal": research_journal(env.state),
        "quality": dict(env.state.quality),
        "trajectory": trajectory_metrics(env.state),
        "resources": dict(env.state.resources),
        "reviews": list(env.state.reviews),
        "project_epoch": int(env.state.project_epoch),
        "output_kind": "scientific_runtime",
        "status_file": str(Path(workspace) / "RESEARCH_STATUS.md"),
        "result_file": str(Path(workspace) / "RESULT.md"),
        "handoff": handoff,
        "handoff_file": (str(Path(workspace) / "RESEARCH_HANDOFF.json") if handoff else ""),
        "controller": "OpenWorld",
        "ticks": env.state.ticks,
        "world_id": env.state.world_id,
        "harness_version": env.harness.version_id,
        "resolved": [row.get("id") for row in env.state.resolved_questions],
        "open": [row.get("id") for row in env.state.open_questions],
        "blocked": [row.get("id") for row in env.state.blocked_questions],
        "execution_gap_reasons": gaps,
        "frontier": dict(env.state.frontier),
        "debt": dict(env.state.debt),
        "metrics": dict(env.state.metrics),
        "speculative_frontier": list(env.state.speculative_frontier),
        "verified_research_state": dict(env.state.verified_research_state),
        "digest": state_digest(env.state),
        "rebuild_digest": state_digest(rebuilt),
        "digest_ok": state_digest(env.state) == state_digest(rebuilt),
        # Whether the work is worth pursuing and whether it has been shown are
        # separate answers; `research_status` above is the second one.
        "discovery": discovery,
        "discovery_status": discovery["discovery_status"],
        "research_promise": discovery["research_promise"],
        "evidence_status": discovery["evidence_status"],
        "deliverables": deliverables,
    }
