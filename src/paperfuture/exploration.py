"""One causal research candidate per round, a model panel, and cumulative navigation."""
from __future__ import annotations

import copy
from .contracts import ContractError, require_refs
from .graph import RUBRIC
from .literature import tokens
from .storage import digest


def validate_proposal(proposal, context, inherited):
    graph=context["complete_stage_graph"]
    require_refs(proposal["source_stages"],{s["id"] for s in graph["stages"]},"leap source stages",True)
    require_refs(proposal["source_edges"],{e["id"] for e in graph["edges"]},"leap source edges")
    require_refs(proposal["evidence_ids"],context["evidence"],"leap grounding",True)
    interval=proposal["target_interval"]
    if not interval["lower"]<=proposal["target_weight"]<=interval["upper"]:
        raise ContractError("proposed magnitude must lie in its uncertainty interval")
    if proposal["target_weight"]>=70 and max(proposal["target_delta"]["mechanism"] or 0,proposal["target_delta"]["assumption"] or 0)<7:
        raise ContractError("high-magnitude leap requires a central mechanism or assumption change")
    papers={e["paper_key"] for e in context["evidence"].values()}
    require_refs([p["paper_key"] for p in proposal["nearest_prior_art"]],papers,"nearest prior art",True)
    require_refs(proposal["minimal_experiment"]["baseline_paper_keys"],papers,"minimal experiment baselines",True)
    ids=[a["id"] for a in proposal["assumptions"]]
    if len(ids)!=len(set(ids)): raise ContractError("duplicate assumption IDs")
    assumptions={a["id"]:a for a in proposal["assumptions"]}
    for a in assumptions.values():
        require_refs(a["evidence_ids"],context["evidence"],"assumption grounding",a["origin"]=="literature")
    for old in inherited:
        if assumptions.get(old["id"])!=old:
            raise ContractError("inherited conditional assumptions must be preserved verbatim with original epistemic status")
    statements=[tuple(tokens(a["statement"])) for a in assumptions.values()]
    if len(statements)!=len(set(statements)):
        raise ContractError("duplicate active assumption statements; retain the original identifier")
    retrieval={r["paper_key"]:r for r in proposal["retrieval_use"]}
    expected=set(context.get("round_retrieval",{}).get("paper_keys",[]))
    if len(retrieval)!=len(proposal["retrieval_use"]) or set(retrieval)!=expected:
        raise ContractError("retrieval_use must audit every dynamically retrieved paper exactly once")
    for key,item in retrieval.items():
        refs={e for e,s in context["evidence"].items() if s["paper_key"]==key}
        require_refs(item["evidence_ids"],refs,"retrieval use evidence",item["disposition"]=="used")


def validate_simulation(result,proposal,context):
    if {s["kind"] for s in result["scenarios"]}!={"success","partial","failure"}:
        raise ContractError("simulation needs success, partial and failure exactly once")
    for s in result["scenarios"]:
        require_refs(s["evidence_ids"],context["evidence"],"simulation evidence")
        require_refs(s["assumption_ids"],{a["id"] for a in proposal["assumptions"]},"simulation assumptions")


def validate_critique(result,proposal,context):
    ids=[o["id"] for o in result["objections"]]
    if len(ids)!=len(set(ids)): raise ContractError("duplicate objection IDs")
    for o in result["objections"]:
        require_refs(o["evidence_ids"],context["evidence"],"objection evidence",o["kind"]=="evidenced_conflict")
        require_refs(o["assumption_ids"],{a["id"] for a in proposal["assumptions"]},"objection assumptions")


def experiences_from_rounds(rounds):
    """Traceable conditional lessons, never inserted into the historical corpus."""
    result=[]
    for r in rounds:
        assumptions=r["proposal"]["assumptions"]
        if r.get("status")=="rejected_duplicate":
            result.append({"id":f"r{r['number']}:proposal_check", "round":r["number"],"kind":"repetition",
                "content":r["proposal_check"]["reason"],"conditions":[],"evidence_ids":[],"assumptions":[],
                "epistemic_status":"candidate_comparison_not_historical_evidence"})
        for member in r.get("simulation",{}).get("members",[]):
            for s in member["result"]["scenarios"]:
                result.append({"id":f"r{r['number']}:simulation:{member['model_role']}:{s['kind']}","round":r["number"],
                    "kind":s["kind"],"model_role":member["model_role"],
                    "content":"Conditions: "+"; ".join(s["conditions"])+". Predictions: "+"; ".join(s["observable_predictions"])+". Next bottleneck: "+s["next_bottleneck"],
                    "assumptions":[a for a in assumptions if a["id"] in s["assumption_ids"]],
                    "conditions":s["conditions"],"evidence_ids":s["evidence_ids"],
                    "epistemic_status":"conditional_simulation_not_evidence"})
        for o in r.get("critique",{}).get("objections",[]):
            result.append({"id":f"r{r['number']}:critique:{o['id']}","round":r["number"],"kind":o["kind"],
                "content":o["claim"]+". Condition: "+o["condition"]+". Proposed repair: "+o["repair"],
                "assumptions":[a for a in assumptions if a["id"] in o["assumption_ids"]],
                "conditions":[o["condition"]],"evidence_ids":o["evidence_ids"],
                "epistemic_status":"model_objection_not_observed_result"})
        for member in r.get("discussion",{}).get("members",[]):
            for i,implication in enumerate(member["result"]["next_leap_implications"]):
                result.append({"id":f"r{r['number']}:discussion:{member['model_role']}:{i}","round":r["number"],"kind":"navigation",
                    "content":implication["implication"],
                    "assumptions":[a for a in assumptions if a["id"] in implication["assumption_ids"]],
                    "evidence_ids":implication["evidence_ids"],
                    "epistemic_status":"conditional_simulation_not_evidence"})
    return result


def experience_prompt(experiences):
    """Lossless API view: shared conditional assumptions appear only once.

    Persisted experiences retain their full dependencies. Repeating the entire
    assumption list for every discussion implication can otherwise exhaust the
    context before four rounds, despite there being few distinct assumptions.
    """
    catalogue={};items=[]
    for experience in experiences:
        ids=[]
        for assumption in experience["assumptions"]:
            key=assumption["id"]
            if key in catalogue and catalogue[key]!=assumption:
                raise ContractError("cumulative experience assumption ID collision")
            catalogue[key]=assumption;ids.append(key)
        items.append({k:v for k,v in experience.items() if k!="assumptions"} | {"assumption_ids":ids})
    return items,catalogue


def validate_assessment(result,critique,context,rounds,experiences=None):
    objections={o["id"]:o for o in critique["objections"]}
    ids=[r["objection_id"] for r in result["resolutions"]]
    if len(ids)!=len(set(ids)) or set(ids)!=set(objections):
        raise ContractError(f"resolve every objection exactly once; missing={sorted(set(objections)-set(ids))}; unexpected={sorted(set(ids)-set(objections))}")
    for name in ("novelty","importance","potential_gain"):
        require_refs(result[name]["evidence_ids"],context["evidence"],name+" score")
    for r in result["resolutions"]:
        o=objections[r["objection_id"]]
        dismissal=o["kind"]=="evidenced_conflict" and (r["disposition"]!="unresolved" or not r["conditions_match"])
        require_refs(r["evidence_ids"],context["evidence"],"objection dismissal/resolution",dismissal)
    parents={r["number"]:r for r in rounds if r.get("status")!="rejected_duplicate"}
    if result["parent_round"] not in parents: raise ContractError("feedback parent must be an existing candidate")
    require_refs(result["preserve_assumption_ids"],{a["id"] for a in parents[result["parent_round"]]["proposal"]["assumptions"]},"feedback inheritance")
    require_refs(result["navigation"]["target_stage_ids"],{s["id"] for s in context["complete_stage_graph"]["stages"]},"navigation stages",True)
    if any(len(q)>400 for q in result["navigation"]["queries"].values()):
        raise ContractError("navigation queries must each be concise (at most 400 characters), not pasted simulation narratives")
    if experiences is not None:
        require_refs(result["navigation"]["experience_ids"],{e["id"] for e in experiences},"navigation experience",bool(experiences))
    require_refs(result["mechanism_audit"]["nearest_prior_art_keys"],{p["key"] for p in context["papers"]},"mechanism comparison",True)
    require_refs(result["mechanism_audit"]["evidence_ids"],context["evidence"],"mechanism audit evidence",True)
    prior={r["number"] for r in rounds[:-1]}
    if result["progress"]["compared_round"] not in (prior if prior else {0}):
        raise ContractError("progress comparison must identify a prior round, or zero for the first")


def score_assessment(assessment,critique,policy):
    raw=[assessment[k]["score"] for k in ("novelty","importance","potential_gain")]
    effective=list(raw); limits=[]
    audit=assessment.get("mechanism_audit",{})
    if audit.get("kind")=="already_known":
        effective[0]=0;limits.append("novelty zero: proposed mechanism is already reported in supplied prior work")
    if audit.get("kind") in ("integration_only","unclear") and effective[0] is not None:
        effective[0]=min(effective[0],2);limits.append("novelty capped: core causal change not established")
    quality=sum(effective)/12 if all(x is not None for x in effective) else None
    resolutions={r["objection_id"]:r for r in assessment["resolutions"]}
    supported=[]
    for objection in critique["objections"]:
        resolution=resolutions[objection["id"]]
        # A missing repeated citation cannot erase an already evidenced conflict.
        if (objection["kind"]=="evidenced_conflict" and objection["evidence_ids"]
                and resolution["disposition"]=="unresolved" and resolution["conditions_match"]):
            supported.append(objection)
    penalty=max((o["severity"] for o in supported),default=0)
    return {"raw_quality":sum(raw)/12 if all(x is not None for x in raw) else None,
        "quality":quality,"effective_novelty":effective[0],"quality_limits":limits,
        "critic_penalty":penalty,"critic_weight":policy["critic_weight"],
        "utility":quality-policy["critic_weight"]*penalty if quality is not None else None,
        "blocking_objection_ids":[o["id"] for o in supported if o["severity"]>=policy["hard_conflict_severity"]],
        "score_semantics":"uncalibrated_ordinal_model_judgment_not_success_probability",
        "conflict_status":"evidence_reference_validated_conditions_judged_by_model"}


def sufficient(round_,policy):
    if not round_.get("assessment") or round_.get("status")=="rejected_duplicate": return False
    a,s=round_["assessment"],round_["scores"]
    return (s["utility"] is not None and s["utility"]>=policy["sufficient_utility"]
        and s["effective_novelty"] is not None and s["effective_novelty"]>=policy["min_novelty"]
        and a["importance"]["score"] is not None and a["importance"]["score"]>=policy["min_importance"]
        and a["mechanism_clear"] and a["validation_path_clear"] and a["mechanism_audit"]["discriminating_test"]
        and a["mechanism_audit"]["kind"] in ("mechanistic_change","justified_interaction")
        and (not a["progress"]["compared_round"] or a["progress"]["mechanism_changed"] or a["progress"]["resolved_previous_issue"])
        and a["feasibility"]=="testable" and not s["blocking_objection_ids"])


def _unchanged(a,b):
    if b.get("status")=="rejected_duplicate": return True
    identical=tokens(a["proposal"]["idea"]["mechanism"])==tokens(b["proposal"]["idea"]["mechanism"])
    progress=b["assessment"]["progress"]
    audited=(progress["compared_round"]==a["number"] and not progress["mechanism_changed"] and not progress["resolved_previous_issue"])
    return (identical and not progress["resolved_previous_issue"]) or audited


def decide(rounds,config):
    last=rounds[-1]
    if sufficient(last,config["policy"]):
        return {"action":"stop_sufficient","terminal":True,"reason":"predeclared mechanism, value and feasibility checks met"}
    if len(rounds)>=config["max_rounds"]:
        return {"action":"stop_limit","terminal":True,"reason":"global candidate-round limit reached"}
    if len(rounds)>=3 and all(_unchanged(rounds[i-1],rounds[i]) for i in (len(rounds)-2,len(rounds)-1)):
        return {"action":"stop_stagnation","terminal":True,"reason":"two consecutive candidate comparisons establish no substantive mechanism or resolved-issue progress"}
    assessed=[r for r in rounds if r.get("assessment")]
    a=assessed[-1]["assessment"];parent=next(r for r in rounds if r["number"]==a["parent_round"])
    action=a["suggested_action"]
    duplicate=len(rounds)>1 and _unchanged(rounds[-2],last)
    if duplicate: action="switch"
    preserve=set(a["preserve_assumption_ids"])
    if action=="continue_leap": preserve.update(x["id"] for x in parent["proposal"]["assumptions"] if x["origin"]=="hypothesis")
    inherited=[x for x in parent["proposal"]["assumptions"] if x["id"] in preserve]
    if action=="continue_leap":
        inherited.append({"id":f"conditional_parent_round_{parent['number']}","statement":f"Conditional working hypothesis, NOT an observed result: candidate {parent['number']} mechanism proves viable under its stated conditions. Mechanism: {parent['proposal']['idea']['mechanism']}","origin":"hypothesis","evidence_ids":[]})
    return {"action":action,"terminal":False,"parent_round":parent["number"],"next_question":a["next_question"],
        "suggested_change":a["suggested_change"],"navigation":copy.deepcopy(a["navigation"]),"duplicate_candidate_detected":duplicate,
        "inherited_assumptions":inherited,"parent_idea":parent["proposal"]["idea"],
        "parent_unknowns":parent["assessment"]["remaining_unknowns"],
        "explored_stage_sets":[r["proposal"]["source_stages"] for r in rounds],
        "required_change":{"action":action,"axis":a["navigation"]["change_axis"],
            "issue":a["navigation"]["unresolved_issue"],"pass_condition":a["navigation"]["pass_condition"],
            "avoid_repeating":[{"round":r["number"],"principle":r["proposal"]["causal_change"]["changed_principle"],
                "intervention":r["proposal"]["minimal_experiment"]["intervention"],"bottleneck":r["proposal"]["bottleneck"]} for r in rounds],
            "controller_overrode_action":duplicate and a["suggested_action"]!="switch"},
        "epistemic_status":"conditional_feedback_not_historical_evidence"}


def validate_proposal_history(value,context,feedback,previous):
    previous=previous or [];experiences=experiences_from_rounds(previous)
    experience_items,experience_assumptions=experience_prompt(experiences)
    inherited=copy.deepcopy(feedback.get("inherited_assumptions",[])) if feedback else []
    catalogue={e["id"]:e for e in experiences}
    archive=dict(experience_assumptions)
    for row in previous:
        for a in row["proposal"]["assumptions"]:
            if a["id"] in archive and archive[a["id"]]!=a: raise ContractError("experience assumption ID collision")
            archive[a["id"]]=a
    for a in inherited: archive[a["id"]]=a
    refs=[e["experience_id"] for e in value["learning_trace"]]
    require_refs(refs,catalogue,"learning trace",bool(previous))
    required={a["id"] for a in inherited} if (feedback or {}).get("action")=="continue_leap" else set()
    considered={a["id"] for a in inherited}
    for key in refs:
        considered.update(a["id"] for a in catalogue[key]["assumptions"])
    considered.update(a["id"] for a in value["assumptions"] if a["id"] in archive)
    decisions={d["assumption_id"]:d for d in value["assumption_decisions"]}
    if len(decisions)!=len(value["assumption_decisions"]) or set(decisions)!=considered:
        raise ContractError(f"assumption_decisions must cover the used experience conditions and inherited/current historical assumptions exactly once: {sorted(considered)}")
    carried={}
    for key,d in decisions.items():
        if key in required and d["disposition"]!="retain":
            raise ContractError("continue_leap depends on its parent's conditional premises; do not silently replace or drop them")
        if d["disposition"]=="retain":
            if d["replacement_ids"]: raise ContractError("retained assumption has no replacement IDs")
            carried[key]=archive[key]
        else:
            if key in {a["id"] for a in value["assumptions"]}:
                raise ContractError("replaced or inapplicable experience conditions must not reappear as active assumptions")
            require_refs(d["replacement_ids"],{a["id"] for a in value["assumptions"]}-set(decisions),
                         "replacement assumptions",d["disposition"]=="replace")
            if d["disposition"]=="not_applicable" and d["replacement_ids"]:
                raise ContractError("not_applicable is not a replacement")
    present={a["id"] for a in value["assumptions"]}
    value["assumptions"].extend(copy.deepcopy(a) for a in carried.values() if a["id"] not in present)
    for a in value["assumptions"]:
        if a["id"] in archive and archive[a["id"]]!=a: raise ContractError("historical assumption ID cannot change meaning or epistemic status")
        if a["id"] not in archive and any(tokens(a["statement"])==tokens(old["statement"]) for old in archive.values()):
            raise ContractError("do not reintroduce an archived assumption under a new ID")
    validate_proposal(value,context,list(carried.values()))


def propose(gateway,context,config,number,feedback,previous=None):
    previous=previous or [];experiences=experiences_from_rounds(previous)
    experience_items,experience_assumptions=experience_prompt(experiences)
    inherited=copy.deepcopy(feedback.get("inherited_assumptions",[])) if feedback else []
    catalogue={e["id"]:e for e in experiences}
    archive=dict(experience_assumptions)
    for row in previous:
        for a in row["proposal"]["assumptions"]:
            if a["id"] in archive and archive[a["id"]]!=a: raise ContractError("experience assumption ID collision")
            archive[a["id"]]=a
    for a in inherited: archive[a["id"]]=a
    def check(value):
        validate_proposal_history(value,context,feedback,previous)
    return gateway.structured("propose",{"question":config["question"],"round":number,"max_rounds":config["max_rounds"],
        "historical_context":context,"conditional_feedback":feedback,"cumulative_experiences":experience_items,
        "experience_assumptions":archive,
        "previous_candidates":[{"number":r["number"],"idea":r["proposal"]["idea"],"causal_change":r["proposal"]["causal_change"],
            "remaining_unknowns":(r.get("assessment") or {}).get("remaining_unknowns",[]),
            "proposal_check":r.get("proposal_check")} for r in previous]},
        "Propose exactly ONE NEW research direction beyond the supplied literature, using the complete stage/edge map and attached evidence. Graph manifests may list archived but unexpanded reference IDs: cite historical claims ONLY from the evidence passages actually attached to this context. Deferred references are pointers, not passages you have read. The historical graph is prior work, NOT a menu of new contributions: replaying an existing edge or proposing a method already described by a later node has zero novelty even if it improves an older baseline. Start from an unresolved limitation of the closest and strongest existing mechanism. nearest_prior_art MUST include that closest existing method, including any method whose name, intervention or reasoning recipe you borrow, not just a weak older comparator. State precisely which currently unsupported causal hypothesis or intervention differs from it. A new comparison of two existing methods alone is not a new defense mechanism. Do not present predicted gains as established observations. Identify a specific bottleneck and a causal principle or assumption to change. A list such as 'add multimodal + online learning + feedback' is not a mechanism: specify the intervention, causal chain, closest prior work, and a matched baseline experiment whose outcome could falsify the claimed advantage. A combination is legitimate only with a precise nontrivial interaction hypothesis and an ablation that distinguishes it from independent components. Do not manufacture novelty from a small evidence pool. Later rounds MUST use concrete cumulative_experiences and learning_trace explaining HOW a previous failure prediction, disagreement or conditional success changes the next location/mechanism. Avoid reformulating the same idea. Follow required_change: revise resolves its stated issue/pass condition; switch must change the causal principle, intervention or bottleneck, NOT merely stage IDs or wording. Predictions remain conditional; never turn simulated success into history. Experience conditions are an immutable ARCHIVE, not automatically active premises. In learning_trace state retain/test/repair/avoid/not_applicable. For each condition of every cited experience, each inherited assumption and each archived ID you use, provide exactly one assumption_decisions record: retain, replace with new active IDs, or not_applicable, with a precise reason. Only retain reactivates a premise. continue_leap MUST retain its required parent dependencies, including the conditional parent hypothesis. On revise/switch, replacing a criticized premise must not resurrect it. Use original IDs for unchanged premises; never duplicate an assumption with a new ID. Only new assumptions get round-prefixed IDs. Record retrieval_use for every dynamically retrieved paper, including justified rejection of irrelevant evidence; used entries require that paper's evidence. origin is exactly literature or hypothesis. " + RUBRIC,
        identity=f"round:{number}:propose",check=check)


def check_candidate(gateway, proposal, previous, feedback, number):
    """Check the ONE reserved candidate; failure consumes its round, never retries a new idea."""
    if not previous:
        return {"accepted":True,"kind":"first_candidate","reason":"first candidate; prior-art novelty is assessed by the panel"}
    def signature(p):
        return tuple(tuple(tokens(x)) for x in (p["idea"]["mechanism"],
            p["minimal_experiment"]["intervention"],p["causal_change"]["changed_principle"],p["bottleneck"]))
    duplicates=[r["number"] for r in previous if signature(r["proposal"])==signature(proposal)]
    if duplicates:
        return {"accepted":False,"kind":"duplicate","compared_rounds":duplicates,"changed_axes":[],
            "reason":"identical causal mechanism, intervention, principle and bottleneck; candidate slot consumed without another simulation"}
    def check(value):
        ids=value["compared_rounds"]
        if len(ids)!=len(set(ids)) or set(ids)!={r["number"] for r in previous}:
            raise ContractError("proposal comparison must cover every earlier candidate")
        if value["kind"]=="duplicate" and value["changed_axes"]:
            raise ContractError("a duplicate has no substantive changed axes")
    result=gateway.structured("proposal_check",{"candidate":proposal,
        "previous_candidates":[{"number":r["number"],"proposal":r["proposal"]} for r in previous],
        "required_change":(feedback or {}).get("required_change"),"action":(feedback or {}).get("action")},
        "Compare this fixed candidate with EVERY earlier candidate BEFORE expensive simulation. Do not generate or repair an idea. Judge causal equivalence, not word overlap or changed graph IDs. changed_axes lists only substantive bottleneck/principle/intervention/test changes. A paraphrase, renamed component, identical intervention at a new graph node, or unchanged mechanism with generic extra modules is duplicate. A precise changed test resolving a named feasibility issue can be substantive_revision for revise; switch needs a substantive bottleneck, principle or intervention change. Mark unclear when unable to establish change; uncertainty is not a novelty veto. Explain what unresolved issue the new intervention actually addresses. This check does not establish global novelty.",
        identity=f"round:{number}:proposal-check",model_role="assess",check=check)
    switch=(feedback or {}).get("action")=="switch"
    accepted=result["kind"]!="duplicate"
    if switch and result["kind"]!="unclear" and not set(result["changed_axes"]) & {"bottleneck","principle","intervention"}:
        accepted=False
    return {**result,"accepted":accepted,"semantics":"model_causal_comparison_not_empirical_evidence"}


def simulate(gateway,context,proposal,number,model_role="simulate"):
    return gateway.structured("simulate",{"proposal":proposal,"historical_context":context},
        "Independently simulate this exact causal mechanism under success, partial and failure conditions. Specify the assumption IDs, causal chain, observable falsifiable predictions, strongest competing explanation and next bottleneck. Distinguish capability from resource/latency effects. No fabricated measurements or probabilities. You have not seen other models' answers. Predictions are conditional, not experiments. Do not generate replacement ideas.",
        identity=f"round:{number}:simulate:{model_role}",model_role=model_role,
        check=lambda v:validate_simulation(v,proposal,context))


def critique(gateway,context,proposal,number,model_role="critique"):
    return gateway.structured("critique",{"proposal":proposal,"historical_context":context},
        "Independently audit the proposed mechanism and test. FIRST compare it with the closest existing method across ALL supplied stages and source passages, not merely the proposer-selected older baselines. If its actual mechanism is already reported, explicitly flag the novelty claim as an evidenced_conflict citing that prior work and explain what remains unchanged; implementing or benchmarking an existing method is not a new mechanism. A historical technical leap is not a future contribution. Preserve unconventional potentially valuable novelty. Distinguish evidenced_conflict (exact evidence and matching conditions), unknown (missing evidence/experiments) and taste. Unknowns do not veto novelty. Identify actual causal failure modes, nearest-work overlap, confounds and constructive repairs, not generic pessimism. You have not seen the simulators. No invented observations/citations/replacement ideas.",
        identity=f"round:{number}:critique:{model_role}",model_role=model_role,
        check=lambda v:validate_critique(v,proposal,context))


def simulate_panel(gateway,context,proposal,number,config):
    return {"epistemic_status":"conditional_simulation_not_evidence","members":[
        {"model_role":role,"model_id":gateway._model(role)["model"],"result":simulate(gateway,context,proposal,number,role)}
        for role in config["world_model"]["simulators"]]}


def critique_panel(gateway,context,proposal,number,config):
    members=[{"model_role":role,"model_id":gateway._model(role)["model"],"result":critique(gateway,context,proposal,number,role)}
             for role in config["world_model"]["critics"]]
    flattened=[{**o,"model_role":m["model_role"],"model_objection_id":o["id"]} for m in members for o in m["result"]["objections"]]
    return {"members":members,"preserved_novelty":"; ".join(m["result"]["preserved_novelty"] for m in members),
        "objections":[{**o,"id":f"C{i+1}"} for i,o in enumerate(flattened)],
        "missed_opportunities":[s for m in members for s in m["result"]["missed_opportunities"]]}


def discuss_panel(gateway,context,row,config):
    if not config["world_model"]["discussion"]: return {"members":[],"status":"disabled_ablation"}
    roles={m["model_role"] for m in row["simulation"]["members"]+row["critique"]["members"]}
    def check(value):
        for d in value["disagreements"]:
            if d["with_model_role"] not in roles: raise ContractError(f"discussion model role must be one of {sorted(roles)}")
        for p in value["revised_predictions"]+value["next_leap_implications"]:
            require_refs(p["evidence_ids"],context["evidence"],"discussion evidence")
            require_refs(p["assumption_ids"],{a["id"] for a in row["proposal"]["assumptions"]},"discussion assumptions")
    members=[]
    for role in config["world_model"]["simulators"]:
        result=gateway.structured("discuss",{"your_model_role":role,"proposal":row["proposal"],"historical_context":context,
            "independent_simulations":row["simulation"],"independent_critiques":row["critique"]},
            "Discuss the independently generated analyses. State agreements and substantive disagreements, revise predictions only when reasoning or evidence warrants it, propose discriminating tests, and derive actionable next-leap implications. For EACH implication cite only its actual assumption_ids and evidence_ids; do not attach every candidate assumption indiscriminately. Do not conform to a majority or treat repeated claims as independent evidence. Preserve promising novelty and unresolved disagreement. No replacement final idea or invented empirical results.",
            identity=f"round:{row['number']}:discussion:{role}",model_role=role,check=check)
        members.append({"model_role":role,"model_id":gateway._model(role)["model"],"result":result})
    return {"members":members,"epistemic_status":"conditional_simulation_not_evidence"}


def assess(gateway,context,round_,previous):
    candidates=previous+[round_];experiences=experiences_from_rounds(candidates)
    experience_items,experience_assumptions=experience_prompt(experiences)
    data={"historical_context":context,"candidate":round_,"cumulative_experiences":experience_items,
        "experience_assumptions":experience_assumptions,
        "previous_candidates":[{"number":r["number"],"proposal":r["proposal"],"assessment":r.get("assessment"),"scores":r.get("scores"),"proposal_check":r.get("proposal_check")} for r in previous]}
    return gateway.structured("assess",data,
        "Synthesize the panel's simulation, criticisms and discussion WITHOUT rewriting the candidate. FIRST audit prior-art overlap across ALL supplied stages and passages, including work omitted from the proposal's nearest_prior_art. Compare against the closest existing mechanism, not only an older weaker baseline. If the proposed mechanism is already reported, set mechanism_audit.kind=already_known and novelty.score=0, cite that paper, and direct the next leap toward its unresolved limitations. A historical edge with high technical-change weight is not a new contribution. Otherwise audit whether the candidate makes a NEW causal mechanism change, establishes a NEW testable interaction, merely integrates modules, or is unclear. Compare nearest prior works explicitly. Value 0..4 anchors: 0 no supported value, 1 weak/local, 2 plausible incremental, 3 substantial with explicit comparative reasons, 4 exceptional with strong specific grounding; null if evidence is inadequate. Lack of a method in supplied papers is NOT proof of global novelty. Assess feasibility separately. An unresolved evidenced conflict still counts if its evidence is not repeated; dismissing an evidenced conflict requires supplied evidence and a concrete condition mismatch or resolution. Missing experiments are unknowns, not contradictions. Report genuine model disagreements rather than voting them away. Audit progress against a prior candidate: changing wording or adding generic modules is not substantive progress. Navigate the next leap using concrete experience IDs and graph stage IDs, with a reason connecting predicted failures/success conditions to that location. navigation must state the unresolved_issue, change_axis, observable pass_condition and THREE short retrieval queries (bottleneck, alternative mechanism, falsifier; at most 400 characters EACH). Do not paste simulation narratives into queries. Same stage IDs can support genuinely different interventions. A switch requires a changed bottleneck/principle/intervention; revise must resolve a named issue. Retain valuable unconventional hypotheses, but never promote a simulation to evidence. parent_round must exist. Do not invent an extra candidate.",
        identity=f"round:{round_['number']}:assess",check=lambda v:validate_assessment(v,round_["critique"],context,candidates,experiences))


def reconcile_candidate(gateway, candidate, rounds, evidence):
    """Apply later findings to an existing candidate without generating a new one."""
    number=candidate["number"]
    objections=[]
    for row in rounds:
        if row["number"]<number: continue
        for objection in row.get("critique",{}).get("objections",[]):
            objections.append({**objection,"id":f"r{row['number']}:{objection['id']}","source_round":row["number"],
                "source_assumptions":[a for a in row["proposal"]["assumptions"] if a["id"] in objection["assumption_ids"]]})
    assumptions={a["id"]:a for a in candidate["proposal"]["assumptions"]}
    def check(value):
        if value["candidate_round"]!=number: raise ContractError("final review cannot change candidate identity")
        resolutions={r["objection_id"]:r for r in value["resolutions"]}
        if len(resolutions)!=len(value["resolutions"]) or set(resolutions)!={o["id"] for o in objections}:
            raise ContractError("final review must explicitly assess every original and later objection exactly once")
        for o in objections:
            r=resolutions[o["id"]]
            dismissal=o["kind"]=="evidenced_conflict" and (r["disposition"]!="unresolved" or not r["conditions_match"])
            require_refs(r["evidence_ids"],evidence,"final objection dismissal/resolution",dismissal)
        findings={a["assumption_id"]:a for a in value["assumption_findings"]}
        if len(findings)!=len(value["assumption_findings"]) or set(findings)!=set(assumptions):
            raise ContractError("final review must distinguish the status of every candidate assumption")
        for a in findings.values():
            require_refs(a["evidence_ids"],evidence,"final assumption finding",a["status"]=="refuted_by_evidence")
        require_refs(value["prior_art_evidence_ids"],evidence,"final prior art",value["prior_art_status"]=="already_known")
    review=gateway.structured("final_review",{"candidate_round":number,"candidate":candidate["proposal"],
        "original_assessment":candidate["assessment"],"objections":objections,
        "later_candidates":[{"number":r["number"],"proposal":r["proposal"],"status":r.get("status","evaluated"),
            "assessment":r.get("assessment"),"proposal_check":r.get("proposal_check")} for r in rounds if r["number"]>number],
        "evidence":evidence},
        "Reassess ONLY this existing candidate in light of later findings; do not generate, improve or rewrite any idea. For every original AND later objection determine applicability to THIS mechanism and its exact conditions, with a reason. A later critique is not automatically applicable to an earlier mechanism. Unknown/taste remain unknown/taste: missing experiments cannot become evidenced conflicts. Dismissing an evidenced conflict needs supplied evidence and an explicit resolution or condition mismatch. Distinguish retained assumptions, assumptions replaced in a later candidate, uncertainty, and actual evidence refuting an assumption. Replacement alone does NOT refute or invalidate an earlier conditional idea. Reconcile feasibility and remaining unknowns. Mark already_known only with comparative prior-art evidence. Scores are not probabilities. Explain why the candidate remains eligible, needs validation, or faces a substantive evidenced obstacle; do not prefer a later candidate simply because it is later.",
        identity=f"final-review:round:{number}",model_role="assess",check=check)
    return {"review":review,"objections":objections}


def apply_final_review(candidate, reconciliation, policy):
    """Keep original round artifacts immutable; rank a separately audited view."""
    row=copy.deepcopy(candidate);review=reconciliation["review"]
    row["original_scores"]=copy.deepcopy(candidate["scores"])
    row["final_review"]=review
    row["critique"]["objections"]=copy.deepcopy(reconciliation["objections"])
    a=row["assessment"]
    a["resolutions"]=copy.deepcopy(review["resolutions"])
    a.update({k:review[k] for k in ("feasibility","feasibility_reason","remaining_unknowns")})
    for finding in review["assumption_findings"]:
        if finding["status"]!="refuted_by_evidence": continue
        identifier="final-assumption:"+finding["assumption_id"]
        row["critique"]["objections"].append({"id":identifier,"claim":finding["reason"],"kind":"evidenced_conflict",
            "severity":1.0,"condition":"candidate depends on this active assumption","evidence_ids":finding["evidence_ids"],
            "assumption_ids":[finding["assumption_id"]],"repair":"requires an independently justified premise change"})
        a["resolutions"].append({"objection_id":identifier,"disposition":"unresolved","conditions_match":True,
            "reason":finding["reason"],"evidence_ids":finding["evidence_ids"]})
    if review["prior_art_status"]=="already_known":
        a["mechanism_audit"].update(kind="already_known",reason=review["prior_art_reason"],evidence_ids=review["prior_art_evidence_ids"])
    elif review["prior_art_status"]=="uncertain":
        a["remaining_unknowns"]=list(dict.fromkeys(a["remaining_unknowns"]+[review["prior_art_reason"]]))
    row["scores"]=score_assessment(a,row["critique"],policy)
    return row
