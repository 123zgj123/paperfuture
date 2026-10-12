"""Transactional orchestration for the six-step PaperFuture pipeline."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .api import ModelGateway
from .config import normalize, public_config
from .exploration import assess, critique, decide, propose, score_assessment, simulate, sufficient
from .graph import build_graph, graph_context
from .literature import LiteratureStore, Snapshot, collect, select_corpus
from .storage import IntegrityError, atomic_json, atomic_text, digest, exclusive, now, read_json, seal, unseal


def implementation_fingerprint():
    import farfield.extras.llm as adapter
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(__file__).parent.glob("*.py"))}
    sources["farfield_chat_adapter"] = Path(adapter.__file__).read_text(encoding="utf-8")
    return digest(sources)


def save_state(directory, state):
    atomic_json(directory / "state.json", {"sha256": digest(state), "payload": state})
    atomic_text(directory / "events.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in state["events"]))


def event(state, name, **details):
    state["events"].append({"sequence": len(state["events"])+1, "at": now(), "event": name, **details})


def render_final(directory, state, config, context, gateway):
    completed = []
    for entry in state["rounds"]:
        path = directory / "rounds" / f"{entry['number']:02}" / "evaluated.json"
        if path.exists():
            completed.append(unseal(path))
    # A recommendation takes precedence over a higher but infeasible raw score.
    def rank(row):
        score = row["scores"]
        return (sufficient(row, config["policy"]), not bool(score["blocking_objection_ids"]),
                row["assessment"]["feasibility"] != "substantive_obstacle",
                score["utility"] if score["utility"] is not None else -2, -row["number"])
    best = max(completed, key=rank) if completed else None
    if state["status"] == "interrupted":
        recommendation = "incomplete"
    elif best is None:
        recommendation = "not_recommended"
    elif sufficient(best, config["policy"]):
        recommendation = "recommend_for_empirical_validation"
    elif best["scores"]["blocking_objection_ids"] or best["assessment"]["feasibility"] == "substantive_obstacle":
        recommendation = "not_recommended"
    else:
        recommendation = "needs_validation"
    final = {"run_id": config["run_id"], "domain": config["domain"], "question": config["question"],
             "snapshot_id": context.get("snapshot_id"), "graph_id": context.get("graph_id"),
             "status": state["status"], "recommendation": recommendation,
             "stop_reason": state.get("stop_reason"), "completed_rounds": len(completed),
             "reserved_rounds": len(state["rounds"]), "selected_round": best["number"] if best else None,
             "candidate": best, "usage": gateway.usage(), "cutoff": config["cutoff"],
             "research_validity": {"model_vintage_evidence": config["model_vintage_evidence"],
                 "claim": "evidence_constrained_generation; model knowledge leakage has not been ruled out",
                 "simulation_is_empirical_evidence": False, "scores_are_calibrated_probabilities": False,
                 "historical_graph_human_verified": False},
             "error": state.get("error")}
    atomic_json(directory / "final_idea.json", final)
    atomic_json(directory / "usage.json", final["usage"])
    lines = [f"# PaperFuture · {config['run_id']}", "", f"研究问题：{config['question']}", "",
             f"结果状态：`{recommendation}`；完成 {len(completed)} 轮 / 全局上限 {config['max_rounds']} 轮。",
             f"停止原因：{state.get('stop_reason') or state['status']}", "",
             "以下价值判断来自基于冻结文献的条件模拟，尚未经真实实验验证；分数不是成功概率。", ""]
    if best:
        idea = best["proposal"]["idea"]
        lines += [f"## {idea['title']}", "", f"**研究问题**：{idea['problem']}", "",
                  f"**核心机制**：{idea['mechanism']}", "", f"**与已有工作的区别**：{idea['difference']}", "",
                  f"**最小验证实验**：{idea['minimal_test']}", "",
                  f"**跃迁依据**：{best['proposal']['historical_pattern']}", "",
                  f"**方向与幅度**：{best['proposal']['operation']}；目标等级 {best['proposal']['target_weight']} / 4。{best['proposal']['magnitude_reason']}", "",
                  "## 价值与可行性", "", "|维度|等级（0–4）|依据|", "|---|---|---|"]
        for name, label in (("novelty", "新颖性"), ("importance", "重要性"), ("potential_gain", "潜在增益")):
            item = best["assessment"][name]
            reason = item["reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"|{label}|{item['score'] if item['score'] is not None else '未知'}|{reason}|")
        lines += ["", f"综合效用：{best['scores']['utility']}；讨论惩罚：{best['scores']['critic_penalty']} × {best['scores']['critic_weight']}。",
                  f"可行性：{best['assessment']['feasibility']}。{best['assessment']['feasibility_reason']}", "",
                  "## 假设与未决问题", ""]
        lines += [f"- [{a['origin']}] {a['statement']}" for a in best["proposal"]["assumptions"]]
        lines += [f"- {x}" for x in best["assessment"]["remaining_unknowns"]]
        lines += ["", "## 讨论记录", ""]
        resolutions = {r["objection_id"]: r for r in best["assessment"]["resolutions"]}
        for objection in best["critique"]["objections"]:
            resolution = resolutions[objection["id"]]
            lines += [f"- {objection['claim']}（{objection['kind']}，{resolution['disposition']}）：{resolution['reason']}"]
        lines += ["", "## 文献证据", ""]
        papers = {p["key"]: p for p in context.get("papers", [])}
        for ref in best["proposal"]["evidence_ids"]:
            if ref not in context.get("evidence", {}):
                continue
            evidence = context["evidence"][ref]
            paper = papers[evidence["paper_key"]]
            lines += [f"- `{ref}` — [{paper['title']}]({paper['source_url']})，{paper['key']}，{evidence['section']} [{evidence['start']}:{evidence['end']}]。"]
    if state.get("error"):
        lines += ["", f"运行中断：{state['error']}。使用同一配置重跑会恢复已保存的轮次。"]
    lines += ["", "完整图、逐轮输入、提案、模拟、独立批评、综合判断与决策均保存在此运行目录。",
              "模型标识和 token 用量按服务端返回记录，未独立验证模型权重身份或训练数据截止时间。", ""]
    atomic_text(directory / "IDEA.md", "\n".join(lines))
    return final


def graph_mermaid(graph):
    # Use only generated integer node names and JSON-escaped labels.
    indices = {s["id"]: f"s{i}" for i,s in enumerate(graph["stages"])}
    lines = ["flowchart LR"]
    for stage in graph["stages"]:
        label = stage["label"].replace('"', "'").replace("\n", " ").replace("[", "(").replace("]", ")")
        lines.append(f'  {indices[stage["id"]]}["{label}"]')
    for edge in graph["edges"]:
        lines.append(f'  {indices[edge["source"]]} -->|{edge["relation"]}: {edge["weight"] or "unknown"}| {indices[edge["target"]]}')
    return "\n".join(lines) + "\n"


def run(config: dict, workspace: Path, *, prepare_only=False, gateway_factory=ModelGateway, progress=None):
    config = normalize(config)
    directory = workspace.resolve() / "runs" / config["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    def report(message):
        if progress:
            progress(message)
    with exclusive(directory / ".lock"):
        spec = public_config(config)
        spec["implementation_fingerprint"] = implementation_fingerprint()
        seal(directory / "spec.json", spec)
        if (directory / "state.json").exists():
            state = unseal(directory / "state.json")
        else:
            state = {"status": "preparing", "phase": "collection", "rounds": [], "feedback": None,
                     "stop_reason": None, "events": []}
            event(state, "run_created", spec_hash=digest(spec))
            save_state(directory, state)
        gateway = gateway_factory(config, directory / "api", digest(spec))
        context = {}
        try:
            snapshot_path = directory / "snapshot.json"
            if not snapshot_path.exists():
                report("1/6 collecting immutable paper versions and freezing the cutoff")
                store = LiteratureStore(workspace / "literature.sqlite")
                try:
                    keys = collect(config, store, workspace / "collection_cache")
                    snapshot = store.freeze(keys, config["cutoff"])
                finally:
                    store.close()
                seal(snapshot_path, snapshot.payload)
            else:
                snapshot = Snapshot(unseal(snapshot_path))
            selected_path = directory / "selected.json"
            if not selected_path.exists():
                state["phase"] = "selection"
                save_state(directory, state)
                report("1/6 retrieving, reranking, extracting and selecting trajectory coverage")
                seal(selected_path, select_corpus(snapshot, config, gateway))
            selected = unseal(selected_path)
            graph_path = directory / "trajectory.json"
            if not graph_path.exists():
                state["phase"] = "graph"
                save_state(directory, state)
                report("2/6 building and validating the weighted directed development graph")
                seal(graph_path, build_graph(snapshot, selected, config, gateway))
            graph = unseal(graph_path)
            context = graph_context(graph, snapshot, selected)
            seal(directory / "graph_context.json", context)
            atomic_text(directory / "trajectory.mmd", graph_mermaid(graph))
            review_path = directory / "trajectory_review.json"
            if not review_path.exists():
                atomic_json(review_path, {"graph_id": graph["graph_id"],
                    "human_review_status": "pending", "checks_passed": ["evidence_membership", "strict_temporal_witnesses", "acyclicity", "selected_paper_coverage"],
                    "coverage_gaps": graph["coverage_gaps"], "edges_requiring_semantic_review": [e["id"] for e in graph["edges"]]})
            terminal_decisions = [r["decision"] for r in state["rounds"] if (r.get("decision") or {}).get("terminal")]
            if terminal_decisions:
                # Rendering may have failed after the terminal decision committed.
                # Recovery must render it, never reserve a fresh candidate.
                state.update(status="complete", phase="final", stop_reason=terminal_decisions[-1]["action"], error=None)
                save_state(directory, state)
            if state["status"] == "complete":
                return render_final(directory, state, config, context, gateway)
            if prepare_only:
                state.update(status="prepared", phase="exploration", error=None)
                save_state(directory, state)
                atomic_json(directory / "usage.json", gateway.usage())
                return {"status": "prepared", "run_id": config["run_id"], "snapshot_id": snapshot.id, "graph_id": graph["graph_id"]}
            state.update(status="running", phase="exploration", error=None, stop_reason=None)
            save_state(directory, state)
            while True:
                undecided = [r for r in state["rounds"] if r.get("decision") is None]
                if undecided:
                    slot = undecided[0]
                elif len(state["rounds"]) < config["max_rounds"]:
                    slot = {"number": len(state["rounds"])+1, "decision": None, "input_feedback": copy.deepcopy(state["feedback"])}
                    state["rounds"].append(slot)
                    event(state, "round_reserved", number=slot["number"])
                    save_state(directory, state)  # reserve globally BEFORE any candidate generation
                else:
                    state.update(status="complete", stop_reason="stop_limit")
                    save_state(directory, state)
                    break
                number = slot["number"]
                round_dir = directory / "rounds" / f"{number:02}"
                seal(round_dir / "context.json", {"graph_id": graph["graph_id"], "snapshot_id": snapshot.id,
                     "feedback": slot["input_feedback"], "question": config["question"]})
                def step(name, function):
                    path = round_dir / f"{name}.json"
                    if not path.exists():
                        report(f"round {number}/{config['max_rounds']}: {name}")
                        seal(path, function())
                    return unseal(path)
                row = {"number": number}
                row["proposal"] = step("proposal", lambda: propose(gateway, context, config, number, slot["input_feedback"]))
                row["simulation"] = step("simulation", lambda: simulate(gateway, context, row["proposal"], number))
                row["critique"] = step("critique", lambda: critique(gateway, context, row["proposal"], number))
                previous = [unseal(directory / "rounds" / f"{r['number']:02}" / "evaluated.json")
                            for r in state["rounds"] if r["number"] < number]
                row["assessment"] = step("assessment", lambda: assess(gateway, context, row, previous))
                row["scores"] = score_assessment(row["assessment"], row["critique"], config["policy"])
                seal(round_dir / "evaluated.json", row)
                decision = decide(previous + [row], config)
                seal(round_dir / "decision.json", decision)
                # One atomic state replacement applies decision + next feedback + terminal flag.
                slot["decision"] = decision
                state["feedback"] = None if decision["terminal"] else decision
                event(state, "round_decided", number=number, action=decision["action"])
                if decision["terminal"]:
                    state.update(status="complete", phase="final", stop_reason=decision["action"])
                save_state(directory, state)
                if decision["terminal"]:
                    break
            report("6/6 rendering an already evaluated idea and its value judgment")
            return render_final(directory, state, config, context, gateway)
        except Exception as exc:
            # Do not include raw HTTP bodies or credential values in user artifacts.
            from urllib.error import HTTPError, URLError
            safe = type(exc).__name__ if isinstance(exc, (HTTPError, URLError)) else str(exc)
            state.update(status="interrupted", stop_reason="incomplete", error=safe)
            event(state, "interrupted", phase=state["phase"], error=safe)
            save_state(directory, state)
            render_final(directory, state, config, context, gateway)
            raise
        finally:
            gateway.close()
