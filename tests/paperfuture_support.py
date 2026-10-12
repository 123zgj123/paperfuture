"""EXPLICITLY SYNTHETIC responses for pipeline regression tests, never research."""
from __future__ import annotations

import io
import json

from paperfuture.config import normalize
from paperfuture.contracts import SCHEMAS, validate


def papers():
    return [{"paper_id": f"fixture:{i}", "version_id": "v1", "title": f"SYNTHETIC agent security stage {i}",
        "first_public_at": f"202{i}-01-01T00:00:00Z", "version_public_at": f"202{i}-01-01T00:00:00Z",
        "source_url": f"https://example.invalid/fixture/{i}",
        "sections": {"abstract": f"Synthetic stage {i}: agent security separates instructions and tool data. This fixture reports a distinct mechanism for stage {i}. Open bottleneck: adaptive attacks."},
        "references": [f"fixture:{i-1}"] if i > 1 else []} for i in (1,2,3)]


def configuration(path, *, run_id="integration", rounds=4, method="hybrid"):
    return normalize({"run_id": run_id, "domain": "agent_security", "question": "How can agent security separate tool data from authorization?",
        "cutoff": "2024-01-01T00:00:00Z", "max_rounds": rounds, "sources": [{"type": "jsonl", "path": str(path)}],
        "retrieval": {"method": method, "candidate_limit": 10, "selected_limit": 3,"hydrate_candidates":0},
        "world_model":{"simulators":["simulate"],"critics":["critique"],"discussion":True,"min_distinct_models":1},
        "models": {"default": {"base_url": "https://fixture.invalid/v1", "model": "synthetic-chat", "api_key_env": "PAPERFUTURE_TEST_KEY"},
                   "embedding": {"base_url": "https://fixture.invalid/v1", "model": "synthetic-embedding", "api_key_env": "PAPERFUTURE_TEST_KEY"}}})


class SyntheticEndpoint:
    def __init__(self, *, high_value=True, stagnate=False, fail_role=None):
        self.requests = []
        self.high_value = high_value
        self.stagnate = stagnate
        self.fail_role = fail_role
        self.failed = False

    def __call__(self, request, timeout=None):
        payload = json.loads(request.data)
        if request.full_url.endswith("/embeddings"):
            self.requests.append(("embedding", payload))
            response = {"data": [{"index": i, "embedding": [1.0, float(len(text)%13+1), 0.5]} for i,text in enumerate(payload["input"])], "usage": {"total_tokens": 10}}
        else:
            # Ignore bounded format-correction suffix for fixture dispatch.
            prompt = payload["messages"][-1]["content"]
            task = json.JSONDecoder().raw_decode(prompt)[0]
            role, data = task["task"], task["input"]
            self.requests.append((role, data))
            if role == self.fail_role and not self.failed:
                self.failed = True
                raise RuntimeError("synthetic one-time transport interruption")
            result = self.respond(role, data)
            validate(result, SCHEMAS[role])
            response = {"model": "synthetic-chat", "choices": [{"message": {"content": json.dumps(result)}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70}}
        return io.BytesIO(json.dumps(response).encode())

    def respond(self, role, data):
        if role == "rerank":
            return {"results": [{"paper_key": p["paper_key"], "relevance": 4, "reason": "synthetic relevant input"} for p in data["papers"]]}
        if role == "extract":
            return {"relevance": 4, "quality": 3, "selection_reason": "fixture covers historical stage", "problem": "tool-data trust",
                    "mechanism": "synthetic mechanism " + data["paper"]["paper_id"], "assumptions": ["tool data may be hostile"],
                    "capabilities": ["fixture capability"], "limitations": ["synthetic abstract only"], "bottlenecks": ["adaptive attacker"],
                    "subtopics": ["trust boundary"], "evidence_ids": list(data["evidence_spans"])}
        if role == "graph_stages":
            nodes, edges = [], []
            rows = sorted(data["papers"], key=lambda p:p["published"])
            for i, p in enumerate(rows):
                x = p["extraction"]
                nodes.append({"id": f"s{i}", "label": f"synthetic stage {i}", "problem": x["problem"], "mechanism": x["mechanism"],
                              "assumptions": x["assumptions"], "capabilities": x["capabilities"], "limitations": x["limitations"],
                              "bottlenecks": x["bottlenecks"], "paper_keys": [p["paper_key"]], "evidence_ids": x["evidence_ids"]})
                if i:
                    a,b = nodes[i-1]["evidence_ids"][0], nodes[i]["evidence_ids"][0]
                    edges.append({"id": f"e{i}", "source": f"s{i-1}", "target": f"s{i}", "relation": "replace_mechanism",
                                  "inherited": "tool-data setting", "changed": "synthetic mechanism", "bottleneck": "adaptive attacker",
                                  "delta": {"problem": 0, "assumption": 1, "mechanism": 2, "capability": 1, "resource": 0},
                                  "rationale": "fixture transition", "confidence": "medium", "source_witness": a, "target_witness": b,
                                  "evidence_ids": [a,b]})
            return {"stages": nodes, "coverage_gaps": ["synthetic evidence, not research"]}
        if role == "graph_edges":
            nodes={s["id"]:s for s in data["stages"]};edges=[]
            for i,(source,target) in enumerate(data["pairs"]):
                refs=[nodes[source]["evidence_ids"][0],nodes[target]["evidence_ids"][0]]
                delta={"problem":2,"assumption":5,"mechanism":7,"capability":5,"resource":2}
                edges.append({"id":f"e{i+1}","source":source,"target":target,"relation":"replace_mechanism",
                    "inherited":"synthetic setting","changed":"synthetic causal mechanism","bottleneck":"adaptive attacker",
                    "delta":delta,"dimension_reasons":{k:{"reason":"synthetic annotated change","evidence_ids":refs} for k in delta},
                    "weight":65,"weight_interval":{"lower":55,"upper":75},"rationale":"fixture transition","confidence":"medium",
                    "source_witness":refs[0],"target_witness":refs[1],"evidence_ids":refs})
            return {"edges":edges,"unsupported_pairs":[]}
        if role == "graph_merge":
            return {"groups":[{"member_stage_ids":[s["id"]],"representative_stage_id":s["id"],"comparisons":[]} for s in data["batch_stages"]]}
        if role == "graph_review":
            delta={"problem":2,"assumption":5,"mechanism":7,"capability":5,"resource":2}
            return {"reviews":[{"edge_id":e["id"],"verdict":"supported","relationship_reason":"synthetic supports inheritance",
                "evidence_ids":e["evidence_ids"],"source_entailment":"synthetic source mechanism","target_entailment":"synthetic changed mechanism",
                "delta":delta,"dimension_reasons":{k:{"reason":"synthetic audit","evidence_ids":e["evidence_ids"]} for k in delta},
                "weight":63,"weight_interval":{"lower":50,"upper":74}} for e in data["edges"]]}
        if role == "propose":
            evidence = list(data["historical_context"]["evidence"])
            inherited = (data.get("conditional_feedback") or {}).get("inherited_assumptions", [])
            return {"source_stages": ["s2"], "source_edges": ["e2"], "bottleneck": "untrusted tool output", "historical_pattern": "boundary refinement",
                    "operation": "change authorization representation", "target_delta": {"problem": 3, "assumption": 7, "mechanism": 7, "capability": 5, "resource": 2},
                    "target_weight": 73,"target_interval":{"lower":60,"upper":85}, "magnitude_reason": "replace a key assumption", "evidence_ids": evidence,
                    "causal_change":{"existing_mechanism":"implicit permission","changed_principle":f"explicit authorization intervention {data['round']}",
                        "causal_chain":["encode intent","constrain tool action"],"why_not_module_stacking":"changes the authorization rule","falsifier":"no improvement at matched utility"},
                    "nearest_prior_art":[{"paper_key":data["historical_context"]["papers"][0]["key"],"shared_elements":"agent execution","mechanism_difference":"explicit authorization"}],
                    "minimal_experiment":{"baseline_paper_keys":[data["historical_context"]["papers"][0]["key"]],"intervention":"change authorization rule","control":"unchanged agent at matched utility",
                        "observables":["unauthorized action rate"],"falsification_rule":"no paired reduction","confound":"over-restricting benign actions"},
                    "learning_trace":[{"experience_id":data["cumulative_experiences"][0]["id"],"implication":"test predicted ambiguity","change_in_this_proposal":"change the explicit permission representation"}] if data["cumulative_experiences"] else [],
                    "assumptions": inherited or [{"id": "a1", "statement": "permission intent can be represented", "origin": "hypothesis", "evidence_ids": []}],
                    "idea": {"title": f"SYNTHETIC idea {data['round']}", "problem": "permission ambiguity", "mechanism": f"test mechanism {data['round']}",
                             "difference": "a fixture-only distinction", "minimal_test": "controlled benign/adversarial paired comparison"}}
        if role == "simulate":
            return {"epistemic_status": "conditional_simulation_not_evidence", "scenarios": [
                {"kind": kind, "conditions": ["hypothesis a1 holds"], "mechanism_chain": ["intent limits tool action"],
                 "observable_predictions": ["compare unauthorized actions with matched utility"], "competing_explanation": "over-restriction",
                 "next_bottleneck": "ambiguous intent", "evidence_ids": [], "assumption_ids": ["a1"]} for kind in ("success","partial","failure")]}
        if role == "critique":
            return {"preserved_novelty": "conditional boundary mechanism", "objections": [{"id": "o1", "claim": "requires experimental evidence",
                "kind": "unknown", "severity": 1.0, "condition": "deployment", "evidence_ids": [], "assumption_ids": ["a1"], "repair": "paired tests"}],
                "missed_opportunities": []}
        if role == "discuss":
            return {"epistemic_status":"conditional_simulation_not_evidence","agreements":["effectiveness remains unknown"],"disagreements":[],
                "revised_predictions":[{"condition":"intent representation is faithful","prediction":"unauthorized actions decrease","evidence_ids":[],"assumption_ids":["a1"]}],
                "discriminating_tests":["matched utility paired test"],"next_leap_implications":["test ambiguity in the permission representation"]}
        if role == "assess":
            number = data["candidate"]["number"]
            score = {"score": 4 if self.high_value else 2, "reason": "synthetic judgment for control-flow testing", "evidence_ids": []}
            return {"novelty": score, "importance": score, "potential_gain": score,
                "feasibility": "testable", "feasibility_reason": "fixture test can be specified", "mechanism_clear": True, "validation_path_clear": True,
                "resolutions": [{"objection_id": o["id"], "disposition": "unresolved", "conditions_match": True,
                                 "reason": "absence of experiments is not a contradiction", "evidence_ids": []} for o in data["candidate"]["critique"]["objections"]],
                "remaining_unknowns": ["empirical effectiveness"], "next_question": "address a further bottleneck", "suggested_action": "continue_leap",
                "suggested_change": "test the conditional next mechanism", "parent_round": number,
                "preserve_assumption_ids": [a["id"] for a in data["candidate"]["proposal"]["assumptions"]],
                "mechanism_signature": "constant" if self.stagnate else f"mechanism{number}", "assumption_signature": "a1 hypothesis",
                "mechanism_audit":{"kind":"mechanistic_change","reason":"causal permission rule is changed", "nearest_prior_art_keys":[data["historical_context"]["papers"][0]["key"]],
                    "evidence_ids":list(data["historical_context"]["evidence"])[:1],"discriminating_test":True},
                "progress":{"compared_round":number-1,"mechanism_changed":not self.stagnate,"resolved_previous_issue":False,"reason":"synthetic controlled progress judgment"},
                "navigation":{"target_stage_ids":["s2"],"experience_ids":[data["cumulative_experiences"][0]["id"]],"rationale":"investigate predicted ambiguity"},
                "model_disagreements":[]}
        raise AssertionError(role)
