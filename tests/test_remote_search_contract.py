"""A failed retrieval must not count as a jump or disappear from model context."""
import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.openworld.remote import (
    ConceptSpaceSearch, RemoteFrame, RemoteSearchResult, SHIPPED_CORPUS, as_remote_result,
)
from farfield.extras.openworld.ideation import ideation_prompt, ideate
from farfield.extras.openworld.discovery import REMOTE_BOOTSTRAP
from test_discovery_runtime import IDEA, ScriptedClient, base_state
from farfield.extras.openworld.actions import eligible_actions, IDEATE


class RetrievalAccounting(unittest.TestCase):
    def test_backend_success_flag_cannot_promote_an_empty_result(self):
        row = as_remote_result({"used": True, "frames": []}, purpose=REMOTE_BOOTSTRAP)
        self.assertTrue(row["attempted"])
        self.assertFalse(row["used"])
        self.assertEqual(row["status"], "no_candidates")

    def test_failed_dataclass_result_is_not_a_success(self):
        row = RemoteSearchResult(purpose=REMOTE_BOOTSTRAP, backend="concept_space",
                                 gap="remote_space_unavailable:missing").to_dict()
        self.assertEqual(row["status"], "failed")
        self.assertFalse(row["used"])

    def test_extra_metadata_cannot_override_measured_status(self):
        row = RemoteSearchResult(purpose=REMOTE_BOOTSTRAP, backend="concept_space",
                                 extra={"used": True, "frames": [{"label": "invented"}]}).to_dict()
        self.assertFalse(row["used"])
        self.assertEqual(row["frames"], [])

    def test_malformed_and_nonfinite_frames_are_not_evidence_of_retrieval(self):
        for frame in ({"label": "x", "distance": "oops"},
                      {"label": "x", "distance": float("nan")},
                      {"label": "x", "degree": float("inf")}, None):
            with self.subTest(frame=frame):
                row = as_remote_result({"frames": [frame]}, purpose=REMOTE_BOOTSTRAP)
                self.assertFalse(row["used"])
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["rejected_frames"], 1)

    def test_good_frames_survive_and_duplicates_do_not_inflate_the_count(self):
        row = as_remote_result({"frames": ["Error budgets", "error budgets", None]},
                               purpose=REMOTE_BOOTSTRAP)
        self.assertTrue(row["used"])
        self.assertEqual(len(row["frames"]), 1)
        self.assertEqual(row["rejected_frames"], 1)

    def test_dataclass_and_mapping_have_the_same_accounting(self):
        result = RemoteSearchResult(purpose=REMOTE_BOOTSTRAP, backend="supplied",
                                    frames=(RemoteFrame("queues", "n1", .8),))
        row = result.to_dict()
        self.assertEqual(row, as_remote_result(row, purpose=REMOTE_BOOTSTRAP))

    def test_failed_search_is_visible_to_the_proposer(self):
        remote = as_remote_result({"backend": "error", "gap": "index_missing"},
                                  purpose=REMOTE_BOOTSTRAP)
        text = ideation_prompt(mode=REMOTE_BOOTSTRAP, root="question", remote=remote)
        self.assertNotIn("Leave that basin", text.split("\n\n", 1)[0])
        prompt = json.loads(text.split("\n\n", 1)[1])
        self.assertEqual(prompt["remote_search_gap"], "index_missing")
        self.assertNotIn("remote_conceptual_frames", prompt)
        self.assertIn("do not invent", prompt["remote_search_instruction"])

    def test_malformed_backend_does_not_abort_the_actual_ideation_path(self):
        state = base_state(world_id="W1")
        action = next(a for a in eligible_actions(state, workers_connected=True, discovery=True)
                      if a.action_type == IDEATE)
        client = ScriptedClient([IDEA])
        result = ideate(action, state.to_dict(), client=client, workspace=None,
                        remote_search=lambda **kw: {"frames": [{"label": "x", "distance": "bad"}]})
        self.assertEqual(result.status, "ideated")
        self.assertFalse(result.events[0].payload["remote_search"]["used"])
        self.assertIn("remote_frames_invalid", client.calls[0][1])


class CorpusSelection(unittest.TestCase):
    def test_explicit_missing_corpus_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "concepts").mkdir()
            (root / "graph.json").write_text("{}")
            (root / "concepts/arxiv_registry.json").write_text(json.dumps({
                "snapshots": [{"snapshot_id": SHIPPED_CORPUS, "full_graph": "graph.json"}]}))
            self.assertEqual(ConceptSpaceSearch(root=root)._resolve_graph()[1], SHIPPED_CORPUS)
            result = ConceptSpaceSearch("missing", root=root)(purpose=REMOTE_BOOTSTRAP,
                                                              intent="research").to_dict()
            self.assertFalse(result["used"])
            self.assertEqual(result["requested_corpus"], "missing")
            self.assertEqual(result["status"], "failed")

    def test_fallback_and_lexical_scope_are_exposed(self):
        result = ConceptSpaceSearch()(purpose=REMOTE_BOOTSTRAP, intent="coding agents").to_dict()
        self.assertEqual(result["corpus_fallback"], result["requested_corpus"] != result["corpus"])
        self.assertIn("not_validated_structural_transfer", result["retrieval_scope"])

    def test_invalid_quantile_bands_are_rejected(self):
        for band in ((.9, .1), (-.1, .8), (.1, 1.1), (float("nan"), .8)):
            with self.assertRaises(ValueError):
                ConceptSpaceSearch(band=band)
