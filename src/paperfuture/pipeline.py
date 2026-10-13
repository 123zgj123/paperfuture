"""Transactional orchestration for the six-step PaperFuture pipeline."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .api import ModelGateway
from .config import normalize, public_config
from .exploration import (apply_final_review, assess, check_candidate, critique_panel, decide, discuss_panel,
                          experiences_from_rounds, propose, reconcile_candidate, score_assessment, simulate_panel, sufficient)
from .graph import build_graph, graph_context
from .literature import LiteratureStore, Snapshot, collect, corpus_scope, hydrate_snapshot, round_context, select_corpus
from .storage import IntegrityError, atomic_json, atomic_text, digest, exclusive, now, read_json, seal, unseal
from .memory import memory_view, round_scope


def implementation_fingerprint():
    import farfield.extras.llm as adapter
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(__file__).parent.glob("*.py"))}
    sources.update({'skill:'+str(p.relative_to(Path(__file__).parent)):p.read_text()
                    for p in sorted((Path(__file__).parent/'skills').glob('*/SKILL.md'))})
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
    eligible=[r for r in completed if r.get("assessment")]
    all_evidence=dict(context.get("evidence",{}));all_papers={p["key"]:p for p in context.get("papers",[])}
    for row in completed:
        path=directory/"rounds"/f"{row['number']:02}"/"evidence_context.json"
        if path.exists():
            ctx=unseal(path);all_evidence.update(ctx["evidence"]);all_papers.update({p["key"]:p for p in ctx["papers"]})
    reconciled=[]
    for row in eligible:
        later=[r for r in completed if r["number"]>row["number"]]
        if state["status"]=="complete" and later:
            path=directory/"final_reviews"/f"{row['number']:02}.json"
            if not path.exists():
                # Only passages actually cited by these candidates/objections
                # enter reconciliation. The full archive remains independently
                # resolvable; simulations are never added as source evidence.
                refs=set()
                def cited(value):
                    if isinstance(value,dict):
                        for v in value.values(): cited(v)
                    elif isinstance(value,list):
                        for v in value: cited(v)
                    elif isinstance(value,str) and value in all_evidence: refs.add(value)
                cited([row]+later)
                seal(path,reconcile_candidate(gateway,row,completed,{r:all_evidence[r] for r in sorted(refs)}))
            row=apply_final_review(row,unseal(path),config["policy"])
        reconciled.append(row)
    # A recommendation takes precedence over a higher but infeasible raw score.
    def rank(row):
        score = row["scores"]
        return (sufficient(row, config["policy"]), not bool(score["blocking_objection_ids"]),
                row["assessment"]["feasibility"] != "substantive_obstacle",
                score["utility"] if score["utility"] is not None else -2, -row["number"])
    best = max(reconciled, key=rank) if reconciled else None
    context={**context,"evidence":all_evidence,"papers":list(all_papers.values())}
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
             "evaluated_rounds":len(eligible),
             "rejected_candidate_rounds":[r["number"] for r in completed if r.get("status")=="rejected_duplicate"],
             "selection_audit":[{"round":r["number"],"utility":r["scores"]["utility"],
                 "blocking_objection_ids":r["scores"]["blocking_objection_ids"],"feasibility":r["assessment"]["feasibility"],
                 "later_findings_reconciled":"final_review" in r,
                 "reason":r.get("final_review",{}).get("selection_reason","latest available assessment; no subsequent findings")}
                 for r in reconciled],
             "candidate": best, "usage": gateway.usage(), "cutoff": config["cutoff"],
             "research_validity": {"model_vintage_evidence": config["model_vintage_evidence"],
                 "claim": "evidence_constrained_generation; model knowledge leakage has not been ruled out",
                 "simulation_is_empirical_evidence": False, "scores_are_calibrated_probabilities": False,
                 "historical_graph_human_verified": False,
                 "graph_semantic_review":"independent_LLM_review",
                 "world_model_roles":config["world_model"],
                 "declared_model_ids":{role:model["model"] for role,model in config["models"].items()}},
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
                  f"**方向与幅度**：{best['proposal']['operation']}；LLM 目标评分 {best['proposal']['target_weight']} / 100，区间 {best['proposal']['target_interval']}。{best['proposal']['magnitude_reason']}", "",
                  "## 价值与可行性", "", "|维度|等级（0–4）|依据|", "|---|---|---|"]
        for name, label in (("novelty", "新颖性"), ("importance", "重要性"), ("potential_gain", "潜在增益")):
            item = best["assessment"][name]
            reason = item["reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"|{label}|{item['score'] if item['score'] is not None else '未知'}|{reason}|")
        lines += ["", f"综合效用：{best['scores']['utility']}；讨论惩罚：{best['scores']['critic_penalty']} × {best['scores']['critic_weight']}。",
                  f"机制审查：{best['assessment']['mechanism_audit']['kind']}。{best['assessment']['mechanism_audit']['reason']}",
                  f"评分约束：{'; '.join(best['scores']['quality_limits']) or '无额外上限'}。",
                  f"可行性：{best['assessment']['feasibility']}。{best['assessment']['feasibility_reason']}", "",
                  "## 假设与未决问题", ""]
        lines += [f"- [{a['origin']}] {a['statement']}" for a in best["proposal"]["assumptions"]]
        lines += [f"- {x}" for x in best["assessment"]["remaining_unknowns"]]
        lines += ["", "## 讨论记录", ""]
        resolutions = {r["objection_id"]: r for r in best["assessment"]["resolutions"]}
        for objection in best["critique"]["objections"]:
            resolution = resolutions[objection["id"]]
            lines += [f"- {objection['claim']}（{objection['kind']}，{resolution['disposition']}）：{resolution['reason']}"]
        lines += ["", "## 多模型模拟与逐轮经验", ""]
        for member in best["simulation"]["members"]:
            lines += [f"- 模拟模型：{member['model_id']}（{member['model_role']}），成功／部分成功／失败均为条件预测。"]
        for lesson in best["proposal"]["learning_trace"]:
            lines += [f"- 使用经验 `{lesson['experience_id']}`：{lesson['implication']}；本轮改变：{lesson['change_in_this_proposal']}"]
        lines += ["", "## 文献证据", ""]
        papers = {p["key"]: p for p in context.get("papers", [])}
        refs=set(best["proposal"]["evidence_ids"])
        for name in ("novelty","importance","potential_gain","mechanism_audit"):
            refs.update(best["assessment"][name]["evidence_ids"])
        for item in best["critique"]["objections"]+best["assessment"]["resolutions"]:
            refs.update(item["evidence_ids"])
        for ref in sorted(refs):
            if ref not in context.get("evidence", {}):
                continue
            evidence = context["evidence"][ref]
            paper = papers[evidence["paper_key"]]
            lines += [f"- `{ref}` — [{paper['title']}]({paper['source_url']})，{paper['key']}，{evidence['section']} [{evidence['start']}:{evidence['end']}]。"]
        if best.get("final_review"):
            lines += ["", "## 后续证据与最终选优", "", best["final_review"]["selection_reason"],
                "原始逐轮评分保持不变；此处按 final_reviews 中的适用条件复核后排序。"]
        for r in final["selection_audit"]:
            lines.append(f"- 第 {r['round']} 轮：效用 {r['utility']}；可行性 {r['feasibility']}；后续发现复核 {r['later_findings_reconciled']}；{r['reason']}")
    if final["rejected_candidate_rounds"]:
        lines += ["", f"重复或未执行必要切换的候选轮次：{final['rejected_candidate_rounds']}；均计入四轮上限，未再花费多模型讨论调用。"]
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
        weight=edge["weight"] if edge["weight"] is not None else "unknown"
        lines.append(f'  {indices[edge["source"]]} -->|{edge["relation"]}: {weight}/100| {indices[edge["target"]]}')
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
        if config['research_skills']['enabled']:
            from .research_skills import skill_manifest
            seal(directory/'skills_manifest.json',skill_manifest())
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
            if state["status"] == "interrupted":
                state.update(status="running" if state["phase"] == "exploration" else "preparing",
                             error=None, stop_reason=None)
                event(state, "run_resumed", phase=state["phase"])
                save_state(directory, state)
            snapshot_path = directory / "snapshot.json"
            if not snapshot_path.exists():
                report("1/6 loading the fixed literature release and freezing the cutoff" if config["literature_mode"] == "frozen"
                       else "1/6 collecting immutable paper versions and freezing the cutoff")
                store = LiteratureStore(workspace / "literature.sqlite")
                try:
                    keys = collect(config, store, workspace / "collection_cache")
                    snapshot = corpus_scope(store.freeze(keys, config["cutoff"]),config)
                    if config["literature_mode"] == "acquire":
                        report("1/6 archiving full-text evidence for temporally balanced retrieval candidates")
                        snapshot, failures=hydrate_snapshot(snapshot,config,workspace/"cache"/"arxiv",gateway)
                        atomic_json(directory/"fulltext_acquisition.json",{"failures":failures})
                    else:
                        atomic_json(directory/"fulltext_acquisition.json",{
                            "mode":"frozen", "network_acquisition":False, "failures":[],
                            "policy":"full-text and missing references are supplemented only during an independent corpus update"})
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
            atomic_json(directory/"literature_update_requests.json",{
                "snapshot_id":snapshot.id, "domain":config["domain"],
                'domains':[config['domain'],*config.get('auxiliary_domains',[])],
                "uncollected_references":selected.get("recall_audit",{}).get("uncollected_references",{}),
                "uncollected_history_seeds":selected.get("recall_audit",{}).get("uncollected_history_seeds",[]),
                "coverage_gaps":selected.get("coverage_gaps",[]),
                "policy":"diagnostic queue for a future corpus release; does not mutate this run"})
            if len(selected["selected_keys"])<2:
                raise ValueError("fewer than two relevant evidence-bearing papers; inspect selected.json funnel before broadening collection")
            if config['auxiliary_domains'] and any(selected['domain_selected_counts'].get(d,0)==0
                    for d in [config['domain'],*config['auxiliary_domains']]):
                raise ValueError('combined-domain selection lacks relevant evidence from a declared domain; inspect selected.json and revise the query or supplement a future corpus release')
            graph_path = directory / "trajectory.json"
            if not graph_path.exists():
                state["phase"] = "graph"
                save_state(directory, state)
                report("2/6 building and validating the weighted directed development graph")
                seal(graph_path, build_graph(snapshot, selected, config, gateway))
            graph = unseal(graph_path)
            if graph.get("graph_status")=="insufficient_supported_transitions":
                raise ValueError("no development edges survived independent evidence review; expand or improve the trajectory corpus")
            context = graph_context(graph, snapshot, selected)
            seal(directory / "graph_context.json", context)
            atomic_text(directory / "trajectory.mmd", graph_mermaid(graph))
            review_path = directory / "trajectory_review.json"
            if not review_path.exists():
                atomic_json(review_path, {"graph_id": graph["graph_id"],
                    "human_review_status": "pending", "semantic_review_status":"independent_LLM_review_completed",
                    "checks_passed": ["evidence_membership", "strict_temporal_witnesses", "acyclicity", "selected_paper_coverage", "per_edge_entailment_review", "dimension_score_evidence"],
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
                previous = [unseal(directory / "rounds" / f"{r['number']:02}" / "evaluated.json")
                            for r in state["rounds"] if r["number"] < number]
                scope=round_scope(config['run_id'],number,slot['input_feedback'],previous)
                seal(round_dir/'memory_scope.json',scope)
                all_experience=experiences_from_rounds(previous)
                experience,memory_index=memory_view(all_experience,scope,slot['input_feedback'])
                seal(round_dir/"experience_memory.json",experience)
                seal(round_dir/'experience_index.json',memory_index)
                current_context=round_context(context,snapshot,config,slot["input_feedback"],experience)
                current_context['memory_scope']=scope
                from .research_skills import evidence_tools
                current_context=evidence_tools(gateway,current_context,snapshot,config,round_dir,slot['input_feedback'])
                seal(round_dir/"evidence_context.json",current_context)
                seal(round_dir / "context.json", {"graph_id": graph["graph_id"], "snapshot_id": snapshot.id,
                     "feedback": slot["input_feedback"], "question": config["question"]})
                def step(name, function):
                    path = round_dir / f"{name}.json"
                    if not path.exists():
                        report(f"round {number}/{config['max_rounds']}: {name}")
                        seal(path, function())
                    return unseal(path)
                row = {"number": number,'scope':scope}
                row["proposal"] = step("proposal", lambda: propose(gateway, current_context, config, number, slot["input_feedback"],previous))
                row["proposal_check"] = step("proposal_check",lambda:check_candidate(gateway,row["proposal"],previous,slot["input_feedback"],number))
                if row["proposal_check"]["accepted"]:
                    row["status"]="evaluated"
                    row["simulation"] = step("simulation", lambda: simulate_panel(gateway, current_context, row["proposal"], number,config))
                    row["critique"] = step("critique", lambda: critique_panel(gateway, current_context, row["proposal"], number,config))
                    row["discussion"] = step("discussion", lambda: discuss_panel(gateway,current_context,row,config))
                    row["assessment"] = step("assessment", lambda: assess(gateway, current_context, row, previous))
                    row["scores"] = score_assessment(row["assessment"], row["critique"], config["policy"])
                else:
                    row["status"]="rejected_duplicate"
                    event(state,"candidate_rejected_before_simulation",number=number,reason=row["proposal_check"]["reason"])
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
        except (Exception, KeyboardInterrupt) as exc:
            # Do not include raw HTTP bodies or credential values in user artifacts.
            from urllib.error import HTTPError, URLError
            safe = type(exc).__name__ if isinstance(exc, (HTTPError, URLError, KeyboardInterrupt)) else str(exc)
            state.update(status="interrupted", stop_reason="incomplete", error=safe)
            event(state, "interrupted", phase=state["phase"], error=safe)
            save_state(directory, state)
            render_final(directory, state, config, context, gateway)
            raise
        finally:
            gateway.close()
