from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paperfuture.api import BudgetExceeded, ModelGateway
from paperfuture.config import normalize
from paperfuture.contracts import ContractError, PROPOSAL, validate
from paperfuture.exploration import score_assessment, validate_proposal
from paperfuture.graph import transition_weight, validate_graph
from paperfuture.literature import LiteratureStore, Snapshot, bm25, collect_arxiv, rrf
from paperfuture.pipeline import run
from paperfuture.storage import IntegrityError, exclusive, read_json, seal, unseal
from paperfuture_support import SyntheticEndpoint, configuration, papers


class PaperFutureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "papers.jsonl"
        self.source.write_text("\n".join(json.dumps(p) for p in papers()))
        self.config = configuration(self.source)
        self.env = patch.dict(os.environ, {"PAPERFUTURE_TEST_KEY": "fixture-not-a-real-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def execute(self, endpoint=None, config=None):
        endpoint = endpoint or SyntheticEndpoint()
        with patch("paperfuture.api.urllib.request.urlopen", endpoint):
            result = run(config or self.config, self.root / "workspace")
        return result, endpoint

    def directory(self):
        return self.root / "workspace/runs/integration"

    def test_whole_hybrid_pipeline_early_stop_and_exact_replay(self):
        result, endpoint = self.execute()
        self.assertEqual(result["completed_rounds"], 1)
        self.assertEqual(result["stop_reason"], "stop_sufficient")
        self.assertEqual(result["recommendation"], "recommend_for_empirical_validation")
        self.assertEqual(result["usage"]["network_attempts"], 11)
        self.assertEqual(len([x for x in endpoint.requests if x[0] == "propose"]), 1)
        critic_input = next(x[1] for x in endpoint.requests if x[0] == "critique")
        self.assertNotIn("simulation", critic_input)
        replay_config = copy.deepcopy(self.config)
        replay_config["api_mode"] = "replay"
        with patch("paperfuture.api.urllib.request.urlopen", side_effect=AssertionError("replay touched network")):
            replay = run(replay_config, self.root / "workspace")
        self.assertEqual(result, replay)
        self.assertTrue((self.directory()/"IDEA.md").exists())

    def test_global_four_rounds_and_no_final_extra_generation(self):
        result, endpoint = self.execute(SyntheticEndpoint(high_value=False))
        self.assertEqual(result["completed_rounds"], 4)
        self.assertEqual(result["reserved_rounds"], 4)
        self.assertEqual(result["stop_reason"], "stop_limit")
        self.assertEqual(sum(role == "propose" for role,_ in endpoint.requests), 4)
        self.assertEqual(sum(role == "assess" for role,_ in endpoint.requests), 4)
        self.assertEqual(result["selected_round"], 1)  # stable tie-breaking
        self.assertEqual(result["recommendation"], "needs_validation")
        second = unseal(self.directory()/"rounds/02/proposal.json")
        inherited = next(a for a in second["assumptions"] if a["id"] == "conditional_parent_round_1")
        self.assertEqual(inherited["origin"], "hypothesis")
        self.assertIn("NOT an observed result", inherited["statement"])

    def test_stagnation_requires_three_completed_rounds(self):
        result, _ = self.execute(SyntheticEndpoint(high_value=False, stagnate=True))
        self.assertEqual(result["completed_rounds"], 3)
        self.assertEqual(result["stop_reason"], "stop_stagnation")

    def test_resume_partial_round_without_new_candidate(self):
        endpoint = SyntheticEndpoint(fail_role="critique")
        with self.assertRaisesRegex(RuntimeError, "one-time"):
            self.execute(endpoint)
        state = unseal(self.directory()/"state.json")
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(len(state["rounds"]), 1)
        result, endpoint = self.execute(endpoint)
        self.assertEqual(result["completed_rounds"], 1)
        self.assertEqual(sum(role == "propose" for role,_ in endpoint.requests), 1)
        self.assertEqual(sum(role == "simulate" for role,_ in endpoint.requests), 1)
        self.assertEqual(result["usage"]["unknown_billing_attempts"], 1)

    def test_resume_after_assessment_before_decision(self):
        with patch("paperfuture.pipeline.decide", side_effect=RuntimeError("synthetic crash before decision")):
            with self.assertRaisesRegex(RuntimeError, "before decision"):
                self.execute()
        endpoint = SyntheticEndpoint()
        result, endpoint = self.execute(endpoint)
        self.assertEqual(endpoint.requests, [])
        self.assertEqual(result["completed_rounds"], 1)

    def test_immutable_version_and_future_content_exclusion(self):
        store = LiteratureStore(self.root/"literature.sqlite")
        self.addCleanup(store.close)
        keys = [store.add(p) for p in papers()]
        future = copy.deepcopy(papers()[0])
        future.update(version_id="v2", version_public_at="2025-01-01T00:00:00Z", sections={"abstract": "FUTURE SECRET"})
        keys.append(store.add(future))
        snapshot = store.freeze(keys, self.config["cutoff"])
        self.assertNotIn("FUTURE SECRET", json.dumps(snapshot.payload))
        with self.assertRaisesRegex(ValueError, "outside"):
            snapshot.get("fixture:1@v2")
        changed = copy.deepcopy(papers()[0])
        changed["sections"]["abstract"] = "changed historical text"
        with self.assertRaisesRegex(ValueError, "immutable"):
            store.add(changed)

    def test_graph_unknown_evidence_future_edge_and_cycle_rejected(self):
        self.execute()
        directory = self.directory()
        graph = unseal(directory/"trajectory.json")
        raw = {k: graph[k] for k in ("stages", "edges", "coverage_gaps")}
        for edge in raw["edges"]:
            edge.pop("weight")
        snapshot = Snapshot(unseal(directory/"snapshot.json"))
        selected = unseal(directory/"selected.json")
        validate_graph(raw, snapshot, selected)
        damaged = copy.deepcopy(raw)
        damaged["stages"][0]["evidence_ids"] = ["invented"]
        with self.assertRaises(ContractError):
            validate_graph(damaged, snapshot, selected)
        damaged = copy.deepcopy(raw)
        edge = damaged["edges"][0]
        edge["source"],edge["target"] = edge["target"],edge["source"]
        edge["source_witness"],edge["target_witness"] = edge["target_witness"],edge["source_witness"]
        with self.assertRaisesRegex(ContractError, "strictly older"):
            validate_graph(damaged, snapshot, selected)

    def test_unknown_criticism_does_not_veto_novelty(self):
        result,_ = self.execute()
        row = result["candidate"]
        self.assertEqual(row["scores"]["critic_penalty"], 0)
        self.assertEqual(row["scores"]["utility"], 1)
        critique = copy.deepcopy(row["critique"])
        objection = critique["objections"][0]
        objection.update(kind="evidenced_conflict", evidence_ids=["checked elsewhere"])
        assessment = copy.deepcopy(row["assessment"])
        assessment["resolutions"][0]["evidence_ids"] = ["checked elsewhere"]
        score = score_assessment(assessment, critique, self.config["policy"])
        self.assertEqual(score["utility"], .8)
        self.assertEqual(score["blocking_objection_ids"], ["o1"])

    def test_context_and_config_are_pinned_on_resume(self):
        self.execute()
        changed = copy.deepcopy(self.config)
        changed["question"] = "different experiment"
        with self.assertRaises(IntegrityError):
            self.execute(config=changed)

    def test_hypotheses_cannot_become_literature_on_next_leap(self):
        result,_ = self.execute()
        proposal = copy.deepcopy(result["candidate"]["proposal"])
        context = unseal(self.directory()/"graph_context.json")
        inherited = copy.deepcopy(proposal["assumptions"])
        proposal["assumptions"][0]["origin"] = "literature"
        proposal["assumptions"][0]["evidence_ids"] = list(context["evidence"])[:1]
        with self.assertRaisesRegex(ContractError, "preserved verbatim"):
            validate_proposal(proposal, context, inherited)

    def test_no_silent_dense_fallback_and_invalid_round_limit(self):
        config = copy.deepcopy(self.config)
        del config["models"]["embedding"]
        with self.assertRaisesRegex(ValueError, "embedding"):
            normalize(config)
        config = copy.deepcopy(self.config)
        config["max_rounds"] = 5
        with self.assertRaisesRegex(ValueError, "GLOBAL"):
            normalize(config)

    def test_weights_have_ordinal_not_embedding_semantics(self):
        delta = {"problem":0,"assumption":0,"mechanism":1,"capability":0,"resource":0}
        self.assertEqual(transition_weight(delta),1)
        delta["mechanism"] = 2
        self.assertEqual(transition_weight(delta),3)
        delta["capability"] = 2
        self.assertEqual(transition_weight(delta),4)
        delta["resource"] = None
        self.assertIsNone(transition_weight(delta))

    def test_artifact_tampering_and_concurrent_run_are_rejected(self):
        path = self.root/"sealed.json"
        seal(path, {"a":1})
        path.write_text(path.read_text().replace('"a": 1', '"a": 2'))
        with self.assertRaises(IntegrityError):
            unseal(path)
        with exclusive(self.root/".lock"):
            with self.assertRaisesRegex(RuntimeError,"another process"):
                with exclusive(self.root/".lock"):
                    pass

    def test_budget_is_durable_and_failure_is_not_free(self):
        config = copy.deepcopy(self.config)
        config["api"]["max_calls"] = 1
        with self.assertRaises(BudgetExceeded):
            self.execute(config=config)
        usage = read_json(self.directory()/"usage.json")
        self.assertEqual(usage["network_attempts"],1)
        with self.assertRaises(BudgetExceeded):
            self.execute(config=config)
        self.assertEqual(read_json(self.directory()/"usage.json")["network_attempts"],1)

    def test_bm25_and_rrf(self):
        self.assertEqual(bm25("agent permission", {"a":"video generation", "b":"agent permission"})[0],"b")
        self.assertEqual(rrf([["a","b"],["b","c"]])[0],"b")

    def test_cycle_with_valid_contribution_dates_is_rejected(self):
        self.execute()
        directory = self.directory()
        graph = unseal(directory/"trajectory.json")
        raw = {k: graph[k] for k in ("stages", "edges", "coverage_gaps")}
        for edge in raw["edges"]:
            edge.pop("weight")
        first, middle, last = raw["stages"]
        first["paper_keys"].extend(last["paper_keys"])
        first["evidence_ids"].extend(last["evidence_ids"])
        raw["stages"] = [first, middle]
        raw["edges"][1]["target"] = first["id"]
        with self.assertRaisesRegex(ContractError, "cycle"):
            validate_graph(raw, Snapshot(unseal(directory/"snapshot.json")), unseal(directory/"selected.json"))

    def test_terminal_decision_survives_rendering_failure(self):
        import paperfuture.pipeline as pipeline
        original = pipeline.render_final
        calls = [0]
        def flaky(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise OSError("synthetic output interruption")
            return original(*args, **kwargs)
        with patch("paperfuture.pipeline.render_final", flaky):
            with self.assertRaisesRegex(OSError, "output interruption"):
                self.execute()
        result, endpoint = self.execute()
        self.assertEqual(result["completed_rounds"], 1)
        self.assertEqual(result["stop_reason"], "stop_sufficient")
        self.assertEqual(endpoint.requests, [])

    def test_validation_repair_is_bounded_and_charged(self):
        class Malformed(SyntheticEndpoint):
            def respond(self, role, data):
                result = super().respond(role, data)
                if role == "propose":
                    result["source_stages"] = ["nonexistent-stage"]
                return result
        endpoint = Malformed()
        with self.assertRaisesRegex(ContractError, "bounded validation"):
            self.execute(endpoint)
        self.assertEqual(sum(role == "propose" for role,_ in endpoint.requests), 2)
        self.assertEqual(len(unseal(self.directory()/"state.json")["rounds"]), 1)
        self.assertEqual(read_json(self.directory()/"final_idea.json")["recommendation"], "incomplete")
        calls = len(endpoint.requests)
        with self.assertRaises(ContractError):
            self.execute(endpoint)
        self.assertEqual(len(endpoint.requests), calls)  # deterministic failed replay, no hidden new proposals

    def test_arxiv_requests_explicit_v1_and_caches_source(self):
        raw = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2401.00001v1</id><title>Versioned paper</title><published>2024-01-01T00:00:00Z</published><updated>2024-01-01T00:00:00Z</updated><summary>Historical abstract</summary></entry></feed>'''
        source = {"type":"arxiv","ids":["2401.00001"]}
        with patch("paperfuture.literature.urllib.request.urlopen", return_value=io.BytesIO(raw)) as network:
            rows = collect_arxiv(source, self.config, self.root/"source_cache")
        self.assertIn("2401.00001v1", network.call_args.args[0].full_url)
        self.assertEqual(rows[0]["version_id"], "v1")
        config = copy.deepcopy(self.config)
        config["api_mode"] = "replay"
        with patch("paperfuture.literature.urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            self.assertEqual(collect_arxiv(source, config, self.root/"source_cache"), rows)

    def test_budget_includes_unknown_transport_attempt_and_recovery_receipt(self):
        config = copy.deepcopy(self.config)
        directory = self.root/"gateway"
        gateway = ModelGateway(config, directory, "fixture-scope")
        self.addCleanup(gateway.close)
        endpoint = SyntheticEndpoint()
        with patch("paperfuture.api.urllib.request.urlopen", endpoint):
            vectors = gateway.embeddings(["synthetic input"], identity="record")
        with gateway.db:
            gateway.db.execute("UPDATE attempts SET status='dispatched',tokens=NULL")
        with patch("paperfuture.api.urllib.request.urlopen", side_effect=AssertionError("receipt should replay")):
            self.assertEqual(gateway.embeddings(["synthetic input"], identity="record"), vectors)
        self.assertEqual(gateway.usage()["network_attempts"],1)
        self.assertEqual(gateway.usage()["unknown_billing_attempts"],0)

    def test_benchmark_rejects_acceptance_year_as_future_proxy(self):
        from paperfuture.bench import audit_cases
        self.execute()
        case = {"case_id":"case1", "run_directory":str(self.directory()),
                "future_papers":[{"paper_id":"future:1","first_public_at":"2025-01-01T00:00:00Z"}],
                "human_judgments":[{"future_paper_id":"future:1","rater_id":"expert1","mechanism_match":3,"rationale":"mechanism matches"}]}
        result = audit_cases([case], self.root)
        self.assertEqual(result["human_future_match_rate"],1)
        self.assertEqual(result["cases"][0]["model_leakage_status"],"unverified")
        case["future_papers"][0]["first_public_at"] = "2023-12-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError,"AFTER"):
            audit_cases([case], self.root)

    def test_missing_embedding_output_is_rejected_without_lexical_fallback(self):
        with patch("paperfuture.api.urllib.request.urlopen", return_value=io.BytesIO(b'{"data":[],"usage":{"total_tokens":1}}')):
            with self.assertRaisesRegex(ContractError,"exactly one"):
                run(self.config, self.root/"workspace")
        self.assertFalse((self.directory()/"selected.json").exists())

    def test_all_domain_identifiers_use_same_pipeline(self):
        for domain in ("multimodal_self_evolution_xai","ai4eda"):
            config = copy.deepcopy(self.config)
            config.update(domain=domain, run_id=domain)
            result,_ = self.execute(config=config)
            self.assertEqual(result["domain"],domain)
            self.assertEqual(result["completed_rounds"],1)

    def test_fulltext_augmentation_does_not_mutate_frozen_abstract(self):
        store = LiteratureStore(self.root/"literature.sqlite")
        self.addCleanup(store.close)
        keys = [store.add(p) for p in papers()]
        old = store.freeze(keys, self.config["cutoff"])
        augmented = copy.deepcopy(papers()[0])
        augmented["sections"]["method"] = "Previously uncollected original method section."
        keys.append(store.add(augmented))
        new = store.freeze(keys, self.config["cutoff"])
        self.assertNotIn("method", old.get("fixture:1@v1")["sections"])
        self.assertIn("method", new.get("fixture:1@v1")["sections"])
        self.assertNotEqual(old.id, new.id)

    def test_concurrent_runs_cannot_import_conflicting_historical_text(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        path = self.root/"shared-literature.sqlite"
        LiteratureStore(path).close()
        barrier = Barrier(2)
        def importer(text):
            store = LiteratureStore(path)
            try:
                record = papers()[0]
                record["sections"]["abstract"] = text
                barrier.wait()
                try:
                    store.add(record)
                    return "accepted"
                except ValueError:
                    return "rejected"
            finally:
                store.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(importer, text) for text in ("first observation", "conflicting observation")]
            self.assertEqual(sorted(f.result() for f in futures), ["accepted","rejected"])

    def test_changed_implementation_requires_new_run(self):
        self.execute()
        with patch("paperfuture.pipeline.implementation_fingerprint", return_value="changed-source"):
            with self.assertRaises(IntegrityError):
                self.execute()

    def test_controller_preserves_hypotheses_even_when_models_omit_them(self):
        class Forgetful(SyntheticEndpoint):
            def respond(self, role, data):
                value = super().respond(role, data)
                if role == "assess":
                    value["preserve_assumption_ids"] = []
                if role == "propose" and data["round"] > 1:
                    value["assumptions"] = []
                return value
        result,_ = self.execute(Forgetful(high_value=False))
        self.assertEqual(result["completed_rounds"], 4)
        fourth = unseal(self.directory()/"rounds/04/proposal.json")
        assumptions = {a["id"]:a for a in fourth["assumptions"]}
        self.assertIn("a1", assumptions)
        self.assertIn("conditional_parent_round_1", assumptions)
        self.assertIn("conditional_parent_round_2", assumptions)
        self.assertIn("conditional_parent_round_3", assumptions)
        self.assertTrue(all(a["origin"] == "hypothesis" for a in assumptions.values()))


if __name__ == "__main__":
    unittest.main()
