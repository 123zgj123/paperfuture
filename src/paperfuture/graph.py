"""Evidence-audited stage DAG; LLM-scored technical changes, not graph distance."""
from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from .api import map_preparation as _map_preparation

from .contracts import ContractError, EDGE, GRAPH, require_refs, validate
from .literature import bm25
from .storage import digest, timestamp


RUBRIC = """Score FIVE different dimensions independently on integer 0..10: problem formulation, assumptions, mechanism, demonstrated capability boundary, resource regime. Anchors: 0 unchanged; 1-2 parameter/local adjustment; 3-4 substantive local redesign; 5-6 structural change; 7-8 change to a central mechanism/assumption or established capability boundary; 9-10 exceptional change of research paradigm. Do not inflate merely for combining named modules or improved metrics. Give each dimension its own evidence IDs and explanation. Mark source_evidence_ids, target_evidence_ids and comparison_evidence_ids within those citations. A single comparative passage can establish both endpoints if its text actually does so; do not require two distinct paper IDs mechanically. Explain what changed relative to the prior mechanism. For resource changes state increase/decrease/tradeoff explicitly: greater cost is not greater research value. null means unknown, not zero. Also directly judge overall technical-change magnitude on integer 0..100, with a model-judgment range containing that judgment; this is NOT a statistical confidence interval. This is an ordinal LLM judgment, NOT calibrated physical distance, probability, importance or citation impact. Explain rather than mechanically summing dimensions. Scores >=70 require an explicit central-mechanism/assumption change; >=90 require demonstrated capability-boundary evidence for historical transitions. Use supplied expert calibration anchors as ordinal comparisons only when the domain and relation type are comparable; explain departures instead of copying their numbers. Anchors are not the heldout evaluation set. For a proposed future leap all gains remain hypothetical."""


def transition_weight(delta):
    """Legacy v0.2 rubric, retained solely for reading historical experiments."""
    if any(x is None for x in delta.values()): return None
    if max(delta["mechanism"], delta["assumption"]) == 2:
        return 4 if delta["capability"] == 2 else 3
    return 2 if max(delta.values()) == 2 or sum(delta.values()) >= 3 else 1


def validate_rating(delta, reasons, weight, interval, evidence):
    if interval["lower"] > interval["upper"] or (weight is not None and not interval["lower"] <= weight <= interval["upper"]):
        raise ContractError("technical score must lie inside its ordered uncertainty interval")
    for name, item in reasons.items():
        require_refs(item["evidence_ids"], evidence, name + " rating", nonempty=delta[name] is not None and delta[name] > 0)
        for field in ("source_evidence_ids","target_evidence_ids","comparison_evidence_ids"):
            require_refs(item[field],item["evidence_ids"],name+" "+field)
        if delta[name] is not None and delta[name]>0 and not (
            item["comparison_evidence_ids"] or (item["source_evidence_ids"] and item["target_evidence_ids"])):
            raise ContractError(f"{name}: positive change needs explicit source/target support or a passage comparing both; IDs alone do not establish entailment")
    if weight is not None and weight >= 70 and max(delta.get("mechanism") or 0, delta.get("assumption") or 0) < 7:
        raise ContractError("overall score >=70 needs a central mechanism/assumption change rated >=7")
    if weight is not None and weight >= 90 and (delta.get("capability") or 0) < 7:
        raise ContractError("overall score >=90 needs capability-boundary evidence rated >=7")


def validate_graph(graph: dict, snapshot, selected: dict) -> None:
    validate(graph, GRAPH)
    stages = {n["id"]: n for n in graph["stages"]}
    if len(stages) != len(graph["stages"]): raise ContractError("duplicate stage IDs")
    keys, used = set(selected["selected_keys"]), set()
    for node in stages.values():
        require_refs(node["paper_keys"], keys, "stage paper keys", True)
        require_refs(node["evidence_ids"], snapshot.evidence, "stage evidence", True)
        if any(snapshot.evidence[e]["paper_key"] not in node["paper_keys"] for e in node["evidence_ids"]):
            raise ContractError("stage evidence must come from its member papers")
        if set(node["paper_keys"]) - {snapshot.evidence[e]["paper_key"] for e in node["evidence_ids"]}:
            raise ContractError("each stage member paper must contribute evidence")
        used.update(node["paper_keys"])
    if used != keys: raise ContractError("every selected paper must be represented")
    dependencies = {key: set() for key in stages}
    edge_ids, pairs = set(), set()
    for edge in graph["edges"]:
        if edge["id"] in edge_ids: raise ContractError("duplicate edge IDs")
        edge_ids.add(edge["id"])
        source, target = edge["source"], edge["target"]
        if source not in stages or target not in stages or source == target:
            raise ContractError("edge endpoints must be distinct known stages")
        if (source,target) in pairs: raise ContractError("duplicate directed stage pair")
        pairs.add((source,target))
        require_refs(edge["evidence_ids"], snapshot.evidence, "edge evidence", True)
        a,b = edge["source_witness"], edge["target_witness"]
        if a not in stages[source]["evidence_ids"] or b not in stages[target]["evidence_ids"]:
            raise ContractError("edge witnesses must belong to their stages")
        if not {a,b} <= set(edge["evidence_ids"]): raise ContractError("edge evidence must include both witnesses")
        if any(snapshot.evidence[e]["paper_key"] not in keys for e in edge["evidence_ids"]):
            raise ContractError("edge evidence outside the selected corpus")
        for reason in edge["dimension_reasons"].values():
            for field,node in (("source_evidence_ids",stages[source]),("target_evidence_ids",stages[target])):
                require_refs(reason[field],snapshot.evidence,"dimension endpoint evidence")
                if any(snapshot.evidence[e]["paper_key"] not in node["paper_keys"] for e in reason[field]):
                    raise ContractError("dimension endpoint evidence belongs to the wrong stage; use comparison_evidence_ids for an explicit cross-paper comparison")
        old,new = (snapshot.get(snapshot.evidence[e]["paper_key"]) for e in (a,b))
        if timestamp(old["version_public_at"]) >= timestamp(new["version_public_at"]):
            raise ContractError("edge requires a strictly older source contribution witness")
        validate_rating(edge["delta"], edge["dimension_reasons"], edge["weight"], edge["weight_interval"], set(edge["evidence_ids"]))
        dependencies[target].add(source)
    try: tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as exc: raise ContractError("technical development graph contains a cycle") from exc


def candidate_pairs(stages, snapshot, predecessors, *, exhaustive_limit=24, sample_size=5, with_audit=False):
    ordered = sorted(stages, key=lambda s:(min(snapshot.get(k)["version_public_at"] for k in s["paper_keys"]),s["id"]))
    reasons={};possible=set()
    def add(source,target,reason):
        reasons.setdefault((source,target),set()).add(reason)
    for i,target in enumerate(ordered):
        older = [n for n in ordered[:i] if min(snapshot.get(k)["version_public_at"] for k in n["paper_keys"])
                 < max(snapshot.get(k)["version_public_at"] for k in target["paper_keys"])]
        if not older: continue
        possible.update((s["id"],target["id"]) for s in older)
        if len(stages)<=exhaustive_limit:
            for n in older:add(n["id"],target["id"],"exhaustive_small_graph")
            continue
        for fields in (("problem","bottlenecks"),("mechanism",),("assumptions","capabilities")):
            def description(n): return " ".join(n[k] if isinstance(n[k],str) else " ".join(n[k]) for k in fields)
            ranked=bm25(description(target),{n["id"]:description(n) for n in older},positive_only=True)
            for source in ranked[:predecessors]:add(source,target["id"],"retrieval:"+"+".join(fields))
        cited = {k for p in target["paper_keys"] for k in snapshot.references(p)}
        add(older[-1]["id"],target["id"],"chronological_neighbor_not_proof_of_relation")
        for n in older:
            if cited & set(n["paper_keys"]):add(n["id"],target["id"],"observed_citation_not_proof_of_inheritance")
    unexamined=possible-set(reasons)
    sampled=sorted(unexamined,key=lambda p:digest(list(p)))[:sample_size]
    for source,target in sampled:add(source,target,"sampled_unrecalled_pair")
    pairs=sorted(reasons)
    audit={"pairs":[{"source":s,"target":t,"reasons":sorted(reasons[s,t])} for s,t in pairs],
           "eligible_pair_count":len(possible),"sampled_unrecalled_pairs":[list(p) for p in sampled],
           "unexamined_pairs":[list(p) for p in sorted(possible-set(reasons))],
           "policy":"earliest-contribution order; strict witness dates and DAG checked separately; no forced connectivity"}
    return (pairs,audit) if with_audit else pairs


def build_graph(snapshot, selected, config, gateway):
    keys = sorted(selected["selected_keys"],key=lambda k:(snapshot.get(k)["version_public_at"],k))
    evidence = {e:span for k in keys for e,span in snapshot.spans(k).items() if e in selected["extractions"][k]["evidence_ids"]}
    opts=config["graph"]; stages=[]; gaps=list(selected["coverage_gaps"]); batch_evidence_gaps=[]
    anchors=config.get("rating_anchors",[]);anchor_evidence={}
    for anchor in anchors:
        source,target=(snapshot.get(anchor[k]) for k in ("source_key","target_key"))
        if timestamp(source["version_public_at"])>=timestamp(target["version_public_at"]):
            raise ContractError("calibration anchor must obey the frozen historical time direction")
        require_refs(anchor["evidence_ids"],snapshot.evidence,"expert anchor evidence",True)
        if {snapshot.evidence[e]["paper_key"] for e in anchor["evidence_ids"]}!={source["key"],target["key"]}:
            raise ContractError("expert anchor evidence must document both endpoint papers")
        anchor_evidence.update({e:snapshot.evidence[e] for e in anchor["evidence_ids"]})
    for offset in range(0,len(keys),opts["stage_batch"]):
        batch=keys[offset:offset+opts["stage_batch"]]
        spans={e:v for e,v in evidence.items() if v["paper_key"] in batch}
        def check(value):
            used=set()
            for node in value["stages"]:
                if len(node["paper_keys"])!=1 or used & set(node["paper_keys"]):
                    raise ContractError("initial stage extraction requires exactly one unique paper per stage; merging requires the separate equivalence audit")
                require_refs(node["paper_keys"],set(batch),"stage members",True)
                require_refs(node["evidence_ids"],spans,"stage evidence",True)
                witnessed={spans[e]["paper_key"] for e in node["evidence_ids"]}
                members=set(node["paper_keys"])
                if witnessed != members:
                    available={k:[e for e,s in spans.items() if s["paper_key"]==k] for k in members}
                    raise ContractError(f"stage {node['id']}: missing paper evidence={sorted(members-witnessed)}; unrelated paper evidence={sorted(witnessed-members)}; valid evidence by member={available}. Keep every input paper represented and correct the evidence IDs.")
                used.update(node["paper_keys"])
            if used != set(batch): raise ContractError(f"stage construction omitted input papers={sorted(set(batch)-used)}; add them to an appropriate stage with their own supplied evidence")
        out=gateway.structured("graph_stages",{"question":config["question"],"papers":[
            {"paper_key":k,"published":snapshot.get(k)["version_public_at"],"extraction":selected["extractions"][k]} for k in batch],"evidence":spans},
            "Extract exactly ONE stage per input paper, with exactly that paper in paper_keys. Do NOT merge papers here, even if the topic is shared; all equivalence merging is audited afterwards. Preserve distinct mechanisms, branches and negative results. Represent every input paper exactly once with exact evidence IDs. Describe demonstrated capabilities separately from speculation, and report missing evidence ONLY within these supplied papers. This is a processing batch, not the entire selected corpus; absence from this batch cannot establish absence from other batches or the field. Stage IDs need only be locally unique; the controller assigns global IDs.",
            identity=f"stages:{snapshot.id}:{offset}",check=check)
        for node in out["stages"]: stages.append({**node,"id":f"s{len(stages)}"})
        if out["coverage_gaps"]:
            # A historical batch can legitimately lack recent evidence even
            # when later batches supply it. Never promote that local statement
            # to a missing year or missing mechanism in the complete graph.
            batch_evidence_gaps.append({"paper_keys":list(batch),
                "scope":"supplied_batch_only_not_global_coverage_or_field_novelty",
                "claims":out["coverage_gaps"]})
    if len(stages)<2: raise ContractError("fewer than two distinct technical stages; broaden trajectory evidence")
    stage_batches=stages
    merged=None
    if len(stages)>1:
        original={s["id"]:s for s in stages}
        def check_merge(value):
            members=[i for group in value["groups"] for i in group["member_stage_ids"]]
            if len(members)!=len(original) or set(members)!=set(original):
                raise ContractError("global stage reconciliation must partition all batch stages exactly once")
            for group in value["groups"]:
                representative=group["representative_stage_id"]
                if representative not in group["member_stage_ids"]:
                    raise ContractError("representative must belong to its original stage group")
                comparisons=group["comparisons"]
                expected=set(group["member_stage_ids"])-{representative}
                if len(comparisons)!=len(expected) or {c["stage_id"] for c in comparisons}!=expected:
                    raise ContractError("compare each merged stage to its representative exactly once; singleton groups have no comparisons")
                for c in comparisons:
                    if not all(c[k] for k in ("same_problem","same_mechanism","same_assumptions","same_capability_boundary")):
                        raise ContractError(f"{c['stage_id']} differs from {representative}; preserve different technical regimes as separate groups")
                    require_refs(c["evidence_ids"],evidence,"stage equivalence evidence",True)
                    witnessed={evidence[e]["paper_key"] for e in c["evidence_ids"]}
                    if not set(original[c["stage_id"]]["paper_keys"]+original[representative]["paper_keys"])<=witnessed:
                        raise ContractError("stage equivalence must cite both member and representative papers")
        merged=gateway.structured("graph_merge",{"batch_stages":stages,"evidence":evidence},
            "Reconcile exact technical regimes across processing batches, NOT broad research topics. Default to singleton groups. Merge ONLY when every member has the SAME specific problem, causal mechanism, operative assumptions AND capability boundary as the representative. For each non-representative, explicitly compare all four criteria and justify equivalence with evidence_ids from BOTH papers. Different methods addressing 'authority boundaries' are NOT the same regime: filtering, re-execution, capability authorization, training, attack search and provenance protocols remain distinct. Do not merge successive technical improvements that should become an edge. Do not summarize a group as a diverse collection of mechanisms. Partition all input stage IDs exactly once. Singletons need an empty comparisons list. Keep reasons concise. The controller preserves original representative descriptions and source evidence; do not rewrite technical content.",
            identity=f"stage-reconciliation:{snapshot.id}",check=check_merge)
        stages=[]
        for group in merged["groups"]:
            nodes=[original[i] for i in group["member_stage_ids"]]
            representative=original[group["representative_stage_id"]]
            stages.append({**representative,"id":f"s{len(stages)}",
                           **{field:list(dict.fromkeys(item for n in nodes for item in n[field]))
                              for field in ("assumptions","capabilities","limitations","bottlenecks")},
                           "paper_keys":sorted({k for n in nodes for k in n["paper_keys"]}),
                           "evidence_ids":sorted({e for n in nodes for e in n["evidence_ids"]})})
    nodes={s["id"]:s for s in stages}
    pairs,pair_audit=candidate_pairs(stages,snapshot,opts["predecessors"],
        exhaustive_limit=opts["exhaustive_stage_limit"],sample_size=opts["unexamined_sample"],with_audit=True)
    proposals=[]; unsupported=[]
    def judge_edges(offset, client):
        batch=pairs[offset:offset+opts["edge_batch"]]; ids={i for p in batch for i in p}
        span_ids={e for n in ids for e in nodes[n]["evidence_ids"]}
        def check(value):
            actual=[(e["source"],e["target"]) for e in value["edges"]+value["unsupported_pairs"]]
            if len(actual)!=len(batch) or set(actual)!=set(batch):
                raise ContractError(f"judge ONLY these pairs exactly once as an edge or unsupported: {batch}; returned {actual}; missing {sorted(set(batch)-set(actual))}")
            for edge in value["edges"]:
                # All citations are explicitly supplied by the model. Aggregate
                # the per-dimension citations instead of requiring a second copy
                # of every ID in an otherwise redundant top-level list.
                refs=set(edge["evidence_ids"])|{edge["source_witness"],edge["target_witness"]}
                refs.update(e for item in edge["dimension_reasons"].values() for e in item["evidence_ids"])
                require_refs([edge["source_witness"]],nodes[edge["source"]]["evidence_ids"],"source_witness must be an evidence quote ID, not a stage ID",True)
                require_refs([edge["target_witness"]],nodes[edge["target"]]["evidence_ids"],"target_witness must be an evidence quote ID, not a stage ID",True)
                allowed=set(nodes[edge["source"]]["evidence_ids"]+nodes[edge["target"]]["evidence_ids"])
                require_refs(refs,allowed,"transition endpoint evidence",True)
                edge["evidence_ids"]=sorted(refs)
            validate_graph({"stages":stages,"edges":value["edges"],"coverage_gaps":[]},snapshot,selected)
        return client.structured("graph_edges",{"pairs":[list(p) for p in batch],"stages":[nodes[i] for i in sorted(ids)],
            "calibration_anchors":anchors,"anchor_evidence":anchor_evidence,
            "witness_candidates":[{"source":s,"target":t,"source_witness_ids":nodes[s]["evidence_ids"],
                                    "target_witness_ids":nodes[t]["evidence_ids"]} for s,t in batch],
            "evidence":{e:evidence[e] for e in span_ids},"dates":{k:snapshot.get(k)["version_public_at"] for n in ids for k in nodes[n]["paper_keys"]}},
            "Judge exactly the supplied earlier/later stage pairs, with one verdict per pair: either an edge OR an unsupported_pairs record. Do not add other pairs. source and target are stage IDs (s0, s1, etc.); source_witness and target_witness are exact evidence aliases E1, E2, etc. chosen from witness_candidates for their respective endpoints, NEVER stage IDs. Create an edge ONLY if supplied target evidence establishes a technical development relation. A citation, similar subject, or a benchmark mentioning a defense does NOT establish inheritance, combination, replacement or demonstrated capability. Reject unsupported pairs explicitly. Cite the supporting spans in each dimension; the controller aggregates these citations. Do not connect a graph merely to make it look complete. " + RUBRIC,
            identity=f"edges:{snapshot.id}:{offset}",check=check)
    for out in _map_preparation(gateway,range(0,len(pairs),opts["edge_batch"]),opts["concurrent_calls"],judge_edges):
        for edge in out["edges"]: proposals.append({**edge,"id":f"e{len(proposals)+1}"})
        unsupported.extend(out["unsupported_pairs"])
    accepted=[]; reviews=[]
    def review_edges(offset, client):
        batch=proposals[offset:offset+opts["review_batch"]]; ids={e["id"] for e in batch}
        spans={e:v for edge in batch for e,v in evidence.items() if e in nodes[edge["source"]]["evidence_ids"]+nodes[edge["target"]]["evidence_ids"]}
        def check(value):
            if len(value["reviews"])!=len(ids) or {r["edge_id"] for r in value["reviews"]}!=ids:
                raise ContractError("independent review must cover every edge exactly once")
            for r in value["reviews"]:
                edge=next(e for e in batch if e["id"]==r["edge_id"])
                allowed=set(nodes[edge["source"]]["evidence_ids"]+nodes[edge["target"]]["evidence_ids"])
                refs=set(r["evidence_ids"])
                refs.update(e for item in r["dimension_reasons"].values() for field in
                    ("evidence_ids","source_evidence_ids","target_evidence_ids","comparison_evidence_ids") for e in item[field])
                r["evidence_ids"]=sorted(refs)
                require_refs(r["evidence_ids"],allowed,"review evidence",r["verdict"]=="supported")
                if r["verdict"]=="supported":
                    if not {edge["source_witness"],edge["target_witness"]} <= set(r["evidence_ids"]):
                        missing={edge["source_witness"],edge["target_witness"]}-set(r["evidence_ids"])
                        raise ContractError(f"review {r['edge_id']}: append missing evidence IDs {sorted(missing)} to the existing evidence_ids list. Both original endpoint quotes must be cited for a supported verdict. Do NOT add source_witness or target_witness fields to the review schema; use the evidence_ids list, or change the verdict to uncertain if these quotes do not support the relation.")
                    validate_rating(r["delta"],r["dimension_reasons"],r["weight"],r["weight_interval"],set(r["evidence_ids"]))
                    reviewed={**edge,**{k:r[k] for k in ("delta","dimension_reasons","weight","weight_interval","evidence_ids",
                        "inherited","changed","bottleneck","rationale","confidence")}}
                    # Endpoint ownership must fail inside the bounded repair
                    # loop, not only after every paid review has completed.
                    validate_graph({"stages":stages,"edges":[reviewed],"coverage_gaps":[]},snapshot,selected)
        return client.structured("graph_review",{"edges":[{k:e[k] for k in
            ("id","source","target","relation","source_witness","target_witness","evidence_ids")} for e in batch],
            "calibration_anchors":anchors,"anchor_evidence":anchor_evidence,
            "stages":[nodes[i] for i in sorted({e[k] for e in batch for k in ("source","target")})],"evidence":spans},
            "Independently audit historical edge ENTAILMENT and technical magnitude. Builder scores, confidence, overall rationale, dimension reasons and change descriptions are hidden. First decide if the proposed relation is actually evidenced, THEN rate magnitude. Explain exactly which source passage demonstrates the starting mechanism and which target passage demonstrates inheritance/change. A chronological comparison of alternative methods does not by itself prove replacement or causal influence. Topic relatedness and citations alone are insufficient. A defense-to-benchmark edge claiming the benchmark incorporates that defense is unsupported unless implementation/use is explicitly evidenced. Use uncertain if evidence is inadequate; only supported relations enter the historical graph. Supply your own inherited/changed/bottleneck, final rationale and confidence, consistent with YOUR scores. Unknown magnitude may be null even when the relationship is supported. Do not invent missing text. " + RUBRIC,
            identity=f"edge-review:{snapshot.id}:{offset}",check=check)
    rating_audit=[]
    for out in _map_preparation(gateway,range(0,len(proposals),opts["review_batch"]),opts["concurrent_calls"],review_edges):
        for r in out["reviews"]:
            edge=next(e for e in proposals if e["id"]==r["edge_id"]);reviews.append(r)
            if r["verdict"]=="supported":
                final={**edge,**{k:r[k] for k in ("delta","dimension_reasons","weight","weight_interval","evidence_ids",
                    "inherited","changed","bottleneck","rationale","confidence")}}
                difference=abs(edge["weight"]-r["weight"]) if edge["weight"] is not None and r["weight"] is not None else None
                disputed=difference is not None and difference>opts["max_rating_disagreement"]
                if disputed:
                    final.update(weight=None,weight_interval={"lower":0,"upper":100},confidence="unknown",
                        rationale=r["rationale"]+" Overall magnitude withheld: independent ratings disagree beyond the declared review threshold; expert adjudication is pending.")
                rating_audit.append({"edge_id":edge["id"],"builder_weight":edge["weight"],"reviewer_weight":r["weight"],
                    "absolute_difference":difference,"magnitude_status":"disputed_unscored" if disputed else "uncalibrated_review_judgment",
                    "final_explanation_source":"reviewer","interval_semantics":"model_judgment_range_not_statistical_confidence_interval",
                    "evidence_levels":sorted({snapshot.get(snapshot.evidence[e]["paper_key"])["content_level"] for e in final["evidence_ids"]})})
                accepted.append(final)
    raw={"stages":stages,"edges":accepted,"coverage_gaps":list(dict.fromkeys(gaps))}
    validate_graph(raw,snapshot,selected)
    result={**raw,"snapshot_id":snapshot.id,"selected_hash":digest(selected),"batch_stages":stage_batches,
        "stage_reconciliation":merged,"edge_reviews":reviews,"batch_evidence_gaps":batch_evidence_gaps,
        "rating_audit":rating_audit,"candidate_pair_audit":pair_audit,
        "calibration_anchors":anchors,"calibration_status":"expert_anchors_supplied_holdout_validation_pending" if anchors else "no_expert_calibration_anchors",
        "proposed_edges":proposals,"unsupported_pairs":unsupported,"candidate_pair_count":len(pairs),
        "unexamined_pair_count":len(pair_audit["unexamined_pairs"]),
        "graph_status":"supported_trajectory" if accepted else "insufficient_supported_transitions",
        "weight_semantics":"LLM_ordinal_0_100_with_five_0_10_dimensions_and_uncertainty_not_probability",
        "evidence_status":"independent_LLM_entailment_review_not_human_verified"}
    result["graph_id"]=digest(result)
    return result


def graph_context(graph, snapshot, selected):
    # Retain every stage/edge in a compact global map. Round-specific passages are
    # retrieved separately, so increasing corpus size does not repeat all PDFs.
    compact={"stages":[dict(s) for s in graph["stages"]],
             "edges":[{k:e[k] for k in ("id","source","target","relation","inherited","changed","bottleneck","delta","weight","weight_interval","evidence_ids")} for e in graph["edges"]],
             "coverage_gaps":graph["coverage_gaps"],
             "batch_evidence_gaps":graph.get("batch_evidence_gaps",[])}
    evidence_ids={e for k in selected["selected_keys"] for e in selected["extractions"][k]["evidence_ids"]}
    # Every paper in the global map needs an actual passage, including older
    # baselines outside this round's retrieval results. Reserve one per paper
    # before spending the remaining passage budget on additional detail.
    pools=[bm25(selected["extractions"][key].get("mechanism","")+" method limitation comparison",
                 {e:snapshot.evidence[e]["quote"] for e in evidence_ids if snapshot.evidence[e]["paper_key"]==key})
           for key in selected["selected_keys"]]
    evidence={pool[0]:snapshot.evidence[pool[0]] for pool in pools if pool}
    budget=max(0,32000-sum(len(e["quote"]) for e in evidence.values()))
    for pool in pools:
        for e in pool[1:]:
            n=len(snapshot.evidence[e]["quote"])
            if n<=budget: evidence[e]=snapshot.evidence[e];budget-=n
    return {"snapshot_id":snapshot.id,"graph_id":graph["graph_id"],"cutoff":snapshot.payload["cutoff"],
        # Release manifests retain their actual totals for provenance, but a
        # retrospective model must see counts only for cutoff-eligible papers.
        'corpus_releases':[{'domain':r['domain'],'release_id':r['release_id'],
            'eligible_papers':sum(r['domain'] in domains for domains in snapshot.payload.get('paper_domains',{}).values())}
            for r in snapshot.payload.get('corpus_releases',[])],
        'paper_domains':{k:snapshot.payload.get('paper_domains',{}).get(k,[]) for k in selected['selected_keys']},
        "complete_stage_graph":compact,"evidence":evidence,
        "evidence_manifest":{"snapshot_id":snapshot.id,"graph_id":graph["graph_id"],
            "all_graph_evidence_ids":sorted({e for n in graph["stages"]+graph["edges"] for e in n.get("evidence_ids",[])}),
            "omitted_passage_ids":sorted(evidence_ids-set(evidence)),
            "resolution":"exact IDs resolve in this run's snapshot.json; target passages expanded during round retrieval"},
        "papers":[{"key":k,"title":snapshot.get(k)["title"],"date":snapshot.get(k)["version_public_at"],
                   "source_url":snapshot.get(k)["source_url"],"content_level":snapshot.get(k)["content_level"]} for k in selected["selected_keys"]],
        "limitations":selected["limitations"]+graph["coverage_gaps"]}
