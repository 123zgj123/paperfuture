"""Discovery and evidence are two runtimes. These tests pin the boundary between them.

Everything here is deterministic: a scripted model double answers by call
purpose, a spy stands in for remote search, and prior states are built from
events. Nothing reaches the network or a live model, and nothing copies the
repository. Test names carry the letter of the requirement they pin (A–Q).
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.llm import Completion
from farfield.extras.openworld import OpenWorld
from farfield.extras.openworld import discovery as D
from farfield.extras.openworld.actions import (
    CRITIQUE,
    EVOLVE_POLICY,
    IDEATE,
    PROBE,
    SPECS,
    SURVEY,
    THEORIZE,
    ActionInstance,
    discovery_actions,
    eligible_actions,
)
from farfield.extras.openworld.events import make_event
from farfield.extras.openworld.ideation import DiscoveryWorkers, ideate
from farfield.extras.openworld.kernel import CandidateResult, readonly_view
from farfield.extras.openworld.reducer import apply_events, rebuild_state, state_digest
from farfield.extras.openworld.runtime import bootstrap_research, run_research
from farfield.extras.openworld.scheduler import research_stage, schedule_all
from farfield.extras.openworld.state import ScientificState

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "how can long-horizon coding agents recover from their own compounding errors?"

IDEA = {
    "title": "Error budgets with verified rollback for long-horizon agents",
    "research_problem": ("Long-horizon coding agents accumulate small errors that compound "
                         "until the task fails."),
    "central_claim": ("Treating residual test failures as a spendable error budget, and rolling "
                      "back to the last verified checkpoint when it is spent, keeps long-horizon "
                      "agents on task."),
    "mechanism": ("An explicit budget turns silent drift into a bounded quantity; rollback "
                  "restores a state whose tests passed, so later steps build on verified work."),
    "motivation": "Agents that fail late waste every earlier step.",
    "why_it_may_matter": "Long tasks are where agents are least reliable and most costly.",
    "novelty_hypothesis": ("Agents retry locally; none treat accumulated error as a budget that "
                           "triggers rollback."),
    "competing_explanation": "Any gain comes from spending more inference compute on retries.",
    "falsifier": ("At matched inference compute, budgeted agents complete no more long tasks "
                  "than unbudgeted ones."),
    "validation_direction": "Compare budgeted and unbudgeted agents at matched compute.",
    "optional_experiment_direction": "Fifty long repository tasks, two arms, tokens matched.",
    "grounding": ["site reliability error budgets"],
    "uncertainties": ["whether test failures track real error"],
    "claim_type": "mechanistic_intervention",
    "structural_correspondence": ("An SRE error budget (allowed unreliability per window) maps "
                                  "onto allowed residual test failures per episode."),
}
IDEA_TWO = {
    **IDEA,
    "title": "Matched-compute error budgets",
    "central_claim": ("At equal inference compute, an error budget with verified rollback "
                      "completes more long tasks than retrying with the same compute."),
    "mechanism": ("The budget decides where compute is spent, returning to verified states "
                  "instead of compounding unverified edits."),
    "falsifier": "With token budgets equalised, the rollback arm does not beat the retry arm.",
    "preserved_from_parent": ["long-horizon setting", "runtime adaptation"],
    "changed_from_parent": ["the causal claim is restated at matched compute"],
    "addresses": "separates the budget mechanism from extra compute",
}
CONCEPTUAL = {
    **IDEA,
    "title": "A taxonomy of compounding agent errors",
    "central_claim": ("Compounding agent errors fall into three classes whose recovery "
                      "strategies differ."),
    "mechanism": ("Errors differ in whether later steps can observe them, which decides "
                  "whether local repair can reach them."),
    "claim_type": "conceptual",
}


def critique_answer(outcome="DEVELOPS", cause="none", *, blocking=False, repairable=True):
    return {
        "rollout": {
            "likely_next_questions": ["does the budget size matter?"],
            "likely_competing_explanations": ["extra compute"],
            "likely_missing_knowledge": ["how often tests miss real errors"],
            "predicted_contradictions": [],
            "required_evidence": ["a matched-compute comparison"],
            "possible_research_path": ["pilot on ten tasks", "matched-compute study",
                                       "ablate the rollback"],
            "dependency_structure": ["a suite of long repository tasks"],
            "likely_failure_point": "tests that miss the errors that compound",
            "what_would_kill_the_idea": "no gain at matched compute",
            "what_result_would_strengthen_it": "a gain that grows with task length",
            "outcome": outcome,
            "reason": "fixture rollout",
        },
        "critique": {
            "strongest_positive_signal": "the long-horizon problem is important",
            "strongest_negative_signal": "the gain could come from more inference compute",
            "what_survives": ["long-horizon problem is important",
                              "runtime adaptation appears useful"],
            "what_failed": [] if cause == "none" else ["the claimed mechanism is not isolated"],
            "root_failure_cause": cause,
            "strongest_competing_explanation": "extra compute explains the improvement",
            "repairable": repairable,
            "blocking": blocking,
            "suggested_operator": "DISCRIMINATIVE_TEST",
            "preserve_constraints": ["long-horizon setting", "runtime adaptation"],
            "change_constraints": ["causal claim", "matched-compute design"],
            "mistakes_to_avoid": ["adding another module without isolating the mechanism"],
            "new_questions": ["what is the right unit for the budget?"],
            "required_evidence": ["a matched-compute comparison"],
            "novelty_assessment": "plausibly new",
            "importance_assessment": "high",
        },
    }


PROMISING = critique_answer()
COMPETING = critique_answer("FRAGILE", "causal_identifiability", blocking=True)
MECHANISM = critique_answer("FRAGILE", "mechanism_unsupported", blocking=True)


class ScriptedClient:
    """Answers by purpose; records every prompt. An unplanned call is a failure."""

    def __init__(self, ideas=(), critiques=()):
        self.ideas, self.critiques, self.calls = list(ideas), list(critiques), []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.calls.append((purpose, prompt))
        if purpose.startswith("discovery_ideate"):
            body = dict(self.ideas.pop(0) if self.ideas else IDEA)
            packet = json.loads(prompt[prompt.index('{'):]).get('current_research_state', {}).get('result_packet', {})
            if packet.get('required'):
                # Exercise the response contract; this scripted receipt is not
                # a scientific judgment or a measure of model capability.
                body['result_response'] = {
                    'packet_digest': packet['packet_digest'], 'disposition': 'continue',
                    'rationale': 'The fixture preserves uncertainty and tests artifact retention.',
                    'next_step': 'Submit the next scripted candidate for validation.',
                    'addressed_results': [dict(result_id=r['result_id'], digest=r['digest'],
                        interpretation='This record does not establish the proposed claim.')
                        for r in packet['results']]}
        elif purpose == "discovery_rollout":
            src = self.critiques[0] if self.critiques else PROMISING
            body = src.get("rollout", src) if isinstance(src, dict) else src
        elif purpose == "discovery_critique":
            src = self.critiques.pop(0) if self.critiques else PROMISING
            body = src.get("critique", src) if isinstance(src, dict) else src
        elif purpose == "retrieval_search_topic":
            body = {"topic": "long horizon coding agent error recovery", "subject": "coding agents"}
        else:
            raise AssertionError(f"unexpected model call: {purpose}")
        return Completion(text=json.dumps(body), model="scripted", request_digest="r",
                          digest="d" * 16, artifact_uri="file:///dev/null", finish_reason="stop",
                          prompt_tokens=1, completion_tokens=1, reasoning_tokens=0, mode="replay")


class SpyRemote:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"backend": "spy",
                "local_basin": [{"label": "agent self-repair", "distance": 0.1}],
                "frames": [{"label": "site reliability error budgets", "distance": 0.71},
                           {"label": "checkpoint and rollback in databases", "distance": 0.66}]}


class EmptyFeed:
    verify_literature = False

    def survey_around(self, *args, **kwargs):
        return []


def base_state(**fields):
    state = apply_events(ScientificState(goal=TOPIC), [
        make_event("ContractSet", {"root_question": TOPIC}),
        make_event("QuestionCreated", {"id": "Q1", "text": TOPIC})])
    for name, value in fields.items():
        setattr(state, name, value)
    return state


def idea_row(idea_id, *, lineage="L1", hop=1, parent="", mode=D.REMOTE_BOOTSTRAP, operator="",
             payload=None, remote=None, distance=0.5):
    used = (mode == D.REMOTE_BOOTSTRAP) if remote is None else remote
    return {"kind": D.IDEA_KIND, "id": idea_id, "lineage_id": lineage, "hop": hop,
            "parent_id": parent, "mode": mode, "operator": operator,
            "policy_version": "exploration-policy-v1", "idea": dict(payload or IDEA),
            "remote_search": ({"used": True, "purpose": mode,
                               "frames": [{"label": f"frame-{idea_id}", "distance": distance}]}
                              if used else {"used": False}),
            "epistemic": "GENERATED"}


def fb_row(idea_id, verdict, cause, *, lineage="L1", hop=1, repairable=True, outcome="FRAGILE"):
    return {"kind": D.FEEDBACK_KIND, "id": f"fb-{idea_id}", "idea_id": idea_id,
            "lineage_id": lineage, "hop": hop, "verdict": verdict, "root_failure_cause": cause,
            "repairable": repairable, "rollout": {"outcome": outcome, "reason": "fixture"},
            "what_failed": ["the claimed mechanism is not isolated"],
            "what_survives": ["long-horizon problem is important"],
            "preserve_constraints": ["long-horizon setting"],
            "structural_validation": [], "epistemic": "GENERATED"}


def with_rows(state, *rows):
    return apply_events(state, [make_event("ArtifactAdded", dict(row)) for row in rows])


def moves(state, action_type, **match):
    return [action for action in discovery_actions(state)
            if action.action_type == action_type
            and all(action.extra.get(key) == value for key, value in match.items())]


class DiscoveryCase(unittest.TestCase):
    def workspace(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def env(self, *rows, client=None, remote=None, adapter=None):
        root = self.workspace()
        env = bootstrap_research(TOPIC, root)
        if rows:
            env.state = apply_events(env.state, [make_event("ArtifactAdded", dict(row))
                                                 for row in rows], log=env.log)
            env.persist()
        env.connect_workers(adapter or DiscoveryWorkers(client=client, workspace=root,
                                                        remote_search=remote))
        return env, root

    def move(self, env, action_type, **match):
        found = moves(env.state, action_type, **match)
        self.assertTrue(found, [(a.action_type, a.extra) for a in discovery_actions(env.state)])
        return found[0]

    def run_project(self, client, horizon=8):
        root = self.workspace()
        spy = SpyRemote()
        events = list(run_research(TOPIC, workspace=root, horizon=horizon, client=client,
                                   feed=EmptyFeed(), remote_search=spy))
        return root, events, spy


class DiscoveryWithoutAWorldTests(DiscoveryCase):
    def test_A_discovery_delivers_research_artifacts_without_an_experimental_world(self):
        root, events, spy = self.run_project(ScriptedClient([IDEA, IDEA_TWO], [COMPETING, PROMISING]))
        done = events[-1]
        self.assertEqual(done["stage"], "done")
        self.assertEqual(done["discovery_status"], D.DISCOVERY_SUCCEEDED)
        for name in ("IDEA.md", "RESEARCH_PLAN.md", "EXPERIMENT_DESIGN.md", "DISCOVERY.md"):
            self.assertTrue((root / name).is_file(), name)
        state = OpenWorld.load(root).state
        self.assertEqual(state.world_versions[0].get("provenance"), "generated")
        self.assertFalse([row for row in state.evidence_records if row.get("epistemic") == "WORLD"])
        self.assertFalse(state.theories, "no registered hypothesis is compiled without a world")
        self.assertFalse(done["research_complete"])
        status = (root / "RESEARCH_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("Promising candidate formed", status)
        self.assertIn("Empirical validation remains incomplete.", status)
        self.assertNotIn("Research failed", status)
        self.assertEqual(spy.calls[0]["purpose"], D.REMOTE_BOOTSTRAP)

    def test_the_failure_case_reads_as_discovery_not_as_a_block(self):
        root, _, _ = self.run_project(ScriptedClient([IDEA, IDEA_TWO], [COMPETING, PROMISING]))
        text = (root / "DISCOVERY.md").read_text(encoding="utf-8")
        for line in (
            "Promising candidate formed",
            "Idea 1 failed because its mechanism was not distinguishable from a competing "
            "explanation.",
            "The failure was archived.",
            "The next hop used DISCRIMINATIVE_TEST and produced Idea 2 without requiring another "
            "latent-space jump.",
            "The resulting research plan and experiment design were preserved.",
            "Empirical validation remains incomplete.",
            "No GENERATED claim was promoted to WORLD.",
        ):
            self.assertIn(line, text)

    def test_B_a_literature_gap_is_recorded_and_the_idea_stands(self):
        root, _, _ = self.run_project(ScriptedClient([IDEA], [PROMISING]), horizon=3)
        state = OpenWorld.load(root).state
        view = D.discovery_view(state)
        self.assertEqual([row["id"] for row in view["ideas"]], ["idea-001"])
        feedback = view["feedback"]["idea-001"]
        self.assertEqual(feedback["literature_grounding"]["status"], "literature_gap")
        self.assertTrue(any("literature" in item for item in feedback["required_evidence"]))
        self.assertEqual(feedback["verdict"], D.PROMISING, "a gap is not a rejection")
        surveys = [row for row in state.artifacts if row.get("kind") == "action_attempt"
                   and row.get("action_type") == SURVEY]
        self.assertEqual(surveys[0]["reason"], "no_verified_literature")

    def test_N_a_later_failed_validation_deletes_no_artifact(self):
        root, _, _ = self.run_project(ScriptedClient([IDEA, CONCEPTUAL], [PROMISING, MECHANISM]),
                                      horizon=10)
        view = D.discovery_view(OpenWorld.load(root).state)
        verdicts = {idea: fb["verdict"] for idea, fb in view["feedback"].items()}
        self.assertEqual(verdicts["idea-001"], D.PROMISING)
        self.assertIn(D.REVISE, verdicts.values(), verdicts)
        first = root / "discovery" / "ideas" / "idea-001"
        for name in ("IDEA.md", "RESEARCH_PLAN.md", "EXPERIMENT_DESIGN.md", "FEEDBACK.json"):
            self.assertTrue((first / name).is_file(), name)
        for name in ("IDEA.md", "RESEARCH_PLAN.md", "EXPERIMENT_DESIGN.md"):
            self.assertTrue((root / name).is_file(), name)
        self.assertEqual(view["statuses"][view["selected_idea"]]["verdict"], D.PROMISING)


class RemoteExplorationTests(DiscoveryCase):
    def test_C_no_trajectory_hit_does_not_stop_exploration(self):
        state = with_rows(base_state(world_id="W1"), {
            "kind": "literature_survey", "id": "s1", "question_id": "Q1",
            "works": [{"title": "Agents that repair their own code", "work_id": "P1"}],
            "research_transitions": []})
        actions = eligible_actions(state, workers_connected=True, discovery=True)
        self.assertFalse([a for a in actions if a.extra.get("research_lane") == "trajectory_supported"])
        boot = [a for a in actions if a.action_type == IDEATE]
        self.assertEqual(boot[0].extra["mode"], D.REMOTE_BOOTSTRAP)
        result = ideate(boot[0], state.to_dict(), client=ScriptedClient([IDEA]), workspace=None,
                        remote_search=SpyRemote())
        self.assertEqual(result.status, "ideated", result.extras)
        row = result.events[0].payload
        self.assertEqual(row["literature"], {"verified_works": 1, "trajectory_hits": 0})

    def test_D_the_first_hop_searches_remotely(self):
        spy, client = SpyRemote(), ScriptedClient([IDEA])
        env, _ = self.env(client=client, remote=spy)
        result = env.execute(self.move(env, IDEATE, mode=D.REMOTE_BOOTSTRAP))
        self.assertEqual(result["status"], "ideated", result)
        self.assertEqual([call["purpose"] for call in spy.calls], [D.REMOTE_BOOTSTRAP])
        row = D.discovery_view(env.state)["ideas"][0]
        self.assertTrue(row["remote_search"]["used"])
        self.assertIn("site reliability error budgets",
                      [frame["label"] for frame in row["remote_search"]["frames"]])
        self.assertIn("remote_conceptual_frames", client.calls[0][1])

    def test_D_the_concept_space_is_deterministic_and_leaves_the_basin(self):
        from farfield.extras.domain import folded_terms
        from farfield.extras.openworld.remote import ConceptSpaceSearch

        search = ConceptSpaceSearch()
        first = search(purpose=D.REMOTE_BOOTSTRAP, intent=TOPIC)
        second = search(purpose=D.REMOTE_BOOTSTRAP, intent=TOPIC)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertFalse(first.gap)
        self.assertTrue(first.frames and first.local_basin)
        basin = set(folded_terms(TOPIC))
        for frame in first.frames:
            self.assertFalse(set(folded_terms(frame.label)) & basin, frame.label)

    def test_E_the_first_hop_must_be_a_complete_idea_not_a_pair(self):
        pair = {"concept_a": "error budgets", "concept_b": "coding agents",
                "pair": ["error budgets", "coding agents"]}
        env, _ = self.env(client=ScriptedClient([pair, IDEA]), remote=SpyRemote())
        action = self.move(env, IDEATE, mode=D.REMOTE_BOOTSTRAP)
        refused = env.execute(action)
        self.assertEqual(refused["status"], "idea_incomplete")
        self.assertFalse(D.discovery_view(env.state)["ideas"])
        accepted = env.execute(action)
        self.assertEqual(accepted["status"], "ideated")
        row = D.discovery_view(env.state)["ideas"][0]
        self.assertEqual(D.idea_of(row).missing(), ())
        for name in D.IDEA_FIELDS:
            self.assertTrue(row["idea"][name], name)

    def test_F_the_second_hop_evolves_from_feedback_without_a_latent_jump(self):
        spy, client = SpyRemote(), ScriptedClient([IDEA_TWO])
        env, _ = self.env(idea_row("idea-001"),
                          fb_row("idea-001", D.REVISE, "competing_explanation_unresolved"),
                          client=client, remote=spy)
        action = self.move(env, IDEATE, mode=D.EVOLVE)
        self.assertEqual(action.extra["operator"], "DISCRIMINATIVE_TEST")
        self.assertFalse(action.extra.get("remote"))
        result = env.execute(action)
        self.assertEqual(result["status"], "ideated", result)
        self.assertEqual(spy.calls, [], "idea evolution must not search the latent space")
        child = D.discovery_view(env.state)["ideas"][-1]
        self.assertFalse(child["remote_search"]["used"])
        self.assertEqual((child["parent_id"], child["addresses_feedback"]),
                         ("idea-001", "fb-idea-001"))
        prompt = client.calls[0][1]
        for text in ('"previous_idea"', '"validation_feedback"',
                     "the claimed mechanism is not isolated", "DISCRIMINATIVE_TEST"):
            self.assertIn(text, prompt)

    def test_G_a_mechanism_failure_is_repaired_not_jumped_away_from(self):
        env, _ = self.env(idea_row("idea-001"), client=ScriptedClient(critiques=[MECHANISM]),
                          remote=SpyRemote())
        result = env.execute(self.move(env, CRITIQUE, idea_id="idea-001"))
        self.assertEqual((result["verdict"], result["root_failure_cause"],
                          result["recommended_next_operator"]),
                         (D.REVISE, "mechanism_unsupported", "MECHANISM_REPAIR"))
        action = self.move(env, IDEATE, parent_idea_id="idea-001")
        self.assertEqual((action.extra["mode"], action.extra["operator"]),
                         (D.EVOLVE, "MECHANISM_REPAIR"))
        self.assertFalse(action.extra.get("remote"))

    def test_H_remote_escape_fires_on_stagnation_and_not_otherwise(self):
        ordinary = with_rows(base_state(), idea_row("idea-001"),
                             fb_row("idea-001", D.REVISE, "mechanism_unsupported"))
        self.assertEqual(moves(ordinary, IDEATE)[0].extra["mode"], D.EVOLVE)
        stuck = with_rows(
            base_state(), idea_row("idea-001"),
            fb_row("idea-001", D.REVISE, "mechanism_unsupported"),
            idea_row("idea-002", hop=2, parent="idea-001", mode=D.EVOLVE,
                     operator="MECHANISM_REPAIR", payload=IDEA_TWO),
            fb_row("idea-002", D.REVISE, "mechanism_unsupported", hop=2),
            idea_row("idea-003", hop=3, parent="idea-002", mode=D.EVOLVE, operator="REFORMULATE",
                     payload=CONCEPTUAL),
            fb_row("idea-003", D.REVISE, "mechanism_unsupported", hop=3))
        escape = moves(stuck, IDEATE)[0]
        self.assertEqual((escape.extra["mode"], escape.extra["operator"]),
                         (D.REMOTE_ESCAPE, "REMOTE_ESCAPE"))
        self.assertTrue(escape.extra["remote"])
        self.assertIn("same_root_failure_mechanism_unsupported_x3", escape.extra["stagnation"])
        dead = with_rows(base_state(), idea_row("idea-001"),
                         fb_row("idea-001", D.COLLAPSE, "unrealistic_assumption", repairable=False))
        self.assertEqual(moves(dead, IDEATE)[0].extra["mode"], D.REMOTE_ESCAPE)

    def test_H_the_escape_searches_remotely_and_avoids_used_frames(self):
        spy = SpyRemote()
        env, _ = self.env(
            idea_row("idea-001"), fb_row("idea-001", D.REVISE, "low_novelty"),
            client=ScriptedClient([IDEA_TWO]), remote=spy)
        action = self.move(env, IDEATE, mode=D.REMOTE_ESCAPE)
        env.execute(action)
        self.assertEqual(spy.calls[0]["purpose"], D.REMOTE_ESCAPE)
        self.assertIn("frame-idea-001", spy.calls[0]["exclude"])

    def test_I_a_closer_but_better_idea_is_preferred(self):
        for distance in (0.01, 0.2):
            state = with_rows(
                base_state(), idea_row("idea-001", distance=0.95),
                fb_row("idea-001", D.REVISE, "competing_explanation_unresolved"),
                idea_row("idea-002", hop=2, parent="idea-001", mode=D.EVOLVE,
                         operator="DISCRIMINATIVE_TEST", payload=IDEA_TWO,
                         remote=distance < 0.1, distance=distance),
                fb_row("idea-002", D.PROMISING, "none", hop=2, outcome="DEVELOPS"))
            view = D.discovery_view(state)
            self.assertEqual(view["selected_idea"], "idea-002")
            self.assertEqual(view["statuses"]["idea-002"]["research_promise"], "HIGH")
            self.assertFalse(moves(state, IDEATE, parent_idea_id="idea-002", mode=D.EVOLVE),
                             "a promising idea is a candidate, not something to push farther")


class ArchiveAndRolloutTests(DiscoveryCase):
    def test_J_ideas_feedback_and_operators_are_archived_and_replay(self):
        client = ScriptedClient([IDEA, IDEA_TWO], [COMPETING, PROMISING])
        env, root = self.env(client=client, remote=SpyRemote())
        env.execute(self.move(env, IDEATE, mode=D.REMOTE_BOOTSTRAP))
        env.execute(self.move(env, CRITIQUE, idea_id="idea-001"))
        env.execute(self.move(env, IDEATE, mode=D.EVOLVE))
        env.execute(self.move(env, CRITIQUE, idea_id="idea-002"))
        kinds = [event.payload.get("kind") for event in env.log.events
                 if event.event_type == "ArtifactAdded"
                 and event.payload.get("kind") in D.DISCOVERY_ARTIFACT_KINDS]
        self.assertEqual(kinds, [D.IDEA_KIND, D.FEEDBACK_KIND, D.IDEA_KIND, D.FEEDBACK_KIND])
        view = D.discovery_view(env.state)
        child = view["ideas"][1]
        self.assertEqual((child["parent_id"], child["addresses_feedback"], child["operator"]),
                         ("idea-001", "fb-idea-001", "DISCRIMINATIVE_TEST"))
        self.assertEqual(view["feedback"]["idea-001"]["verdict"], D.REVISE)
        for idea in ("idea-001", "idea-002"):
            for name in ("IDEA.md", "FEEDBACK.json", "ROLLOUT.json", "RESEARCH_PLAN.md",
                         "EXPERIMENT_DESIGN.md"):
                self.assertTrue((root / "discovery" / "ideas" / idea / name).is_file(), name)
        rebuilt = rebuild_state(env.log)
        self.assertEqual(state_digest(rebuilt), state_digest(env.state))
        self.assertEqual(D.discovery_view(rebuilt), view)
        self.assertEqual(D.discovery_view(OpenWorld.load(root).state), view)

    def test_K_the_research_world_rollout_is_generated_and_classified(self):
        odd = critique_answer("SOMETIMES")
        env, _ = self.env(idea_row("idea-001"), client=ScriptedClient(critiques=[odd]),
                          remote=SpyRemote())
        env.execute(self.move(env, CRITIQUE, idea_id="idea-001"))
        feedback = D.discovery_view(env.state)["feedback"]["idea-001"]
        rollout = feedback["rollout"]
        self.assertIn(rollout["outcome"], D.OUTCOMES)
        self.assertEqual((rollout["outcome"], rollout["outcome_normalized_from"]),
                         (D.FRAGILE, "SOMETIMES"))
        self.assertTrue(rollout["reason"])
        self.assertEqual((rollout["epistemic"], feedback["epistemic"]), ("GENERATED", "GENERATED"))
        self.assertFalse(env.state.evidence_records)
        state = base_state()
        row = idea_row("idea-009")
        for outcome, repairable, verdict in ((D.DEVELOPS, True, D.PROMISING),
                                             (D.COLLAPSES, False, D.REVISE),
                                             (D.COLLAPSES, True, D.REVISE)):
            packet = D.compose_feedback(
                idea_row=row, structural=[], grounding={"status": "literature_gap"},
                rollout=D.normalize_rollout({"outcome": outcome, "reason": "fixture"}),
                critique={"repairable": repairable, "blocking": not repairable,
                          "root_failure_cause": "none" if repairable else "unrealistic_assumption"},
                policy=D.DEFAULT_POLICY, pairs=[], state=state)
            self.assertEqual(packet["verdict"], verdict, outcome)
            if outcome == D.COLLAPSES and not repairable:
                self.assertFalse(packet['critic_repairable'])
                self.assertEqual(packet['investment']['decision'], 'investigate')
                self.assertTrue(packet['repairable'])


class EvidenceRoutingTests(DiscoveryCase):
    def routed_state(self):
        return with_rows(
            apply_events(base_state(), [make_event("WorldCreated", {
                "id": "W1", "world_id": "W1", "digest": "frozen", "attested": True})]),
            {"kind": "literature_survey", "id": "s1", "question_id": "Q1",
             "works": [{"title": "source", "work_id": "P1"}]},
            idea_row("idea-001", payload=CONCEPTUAL),
            fb_row("idea-001", D.PROMISING, "none", outcome="DEVELOPS"),
            idea_row("idea-002", lineage="L2", payload=IDEA),
            fb_row("idea-002", D.PROMISING, "none", lineage="L2", outcome="DEVELOPS"))

    def bridges(self, state):
        return [a.extra["discovery_idea_id"] for a in eligible_actions(state, workers_connected=True)
                if a.action_type == THEORIZE and a.extra.get("research_lane") == "discovery_bridge"]

    def test_L_an_ordinary_claim_skips_the_two_arm_path(self):
        state = self.routed_state()
        self.assertNotIn("idea-001", self.bridges(state))
        statuses = D.discovery_view(state)["statuses"]["idea-001"]
        self.assertEqual((statuses["evidence_route"], statuses["evidence_status"]),
                         ("NOT_APPLICABLE", "NOT_EMPIRICAL"))

    def test_M_a_mechanistic_claim_on_an_attested_world_enters_v4(self):
        state = self.routed_state()
        self.assertEqual(self.bridges(state), ["idea-002"])
        self.assertEqual(SPECS[PROBE].validator, "fair_two_arm")
        compiled = apply_events(state, [make_event("TheoryCreated", {
            "id": "H-idea-002", "question_id": "Q1", "text": IDEA["central_claim"],
            "discovery_idea_id": "idea-002"})])
        self.assertEqual(self.bridges(compiled), [])
        self.assertEqual(D.discovery_view(compiled)["statuses"]["idea-002"]["evidence_status"],
                         "EVIDENCE_PENDING")
        weakened = apply_events(compiled, [make_event("EvidenceAdded", {
            "evidence_id": "E1", "theory_id": "H-idea-002", "epistemic": "WORLD",
            "attested": True, "promotion_ok": True, "outcome": "weakens"})])
        self.assertEqual(D.discovery_view(weakened)["statuses"]["idea-002"]["evidence_status"],
                         "EMPIRICALLY_WEAKENED")

    def test_M_the_bridge_compiles_the_named_idea(self):
        from test_openworld_workers import THEORY, ReplayClient

        from farfield.extras.openworld.workers import LegacyResearchWorkers

        state = ScientificState(goal="query costs for skewed table records", world_id="W1")
        state.add_question(question_id="Q1", text="inference")
        state.artifacts.append({"id": "s1", "kind": "literature_survey", "question_id": "Q1",
                                "works": [{"title": "source", "work_id": "P1"}],
                                "research_transitions": [{"operator": "representation_change",
                                                          "status": "extracted"}]})
        state.artifacts.append(idea_row("idea-001", payload=IDEA))
        state.artifacts.append(fb_row("idea-001", D.PROMISING, "none", outcome="DEVELOPS"))
        metadata = {
            "assumptions": ["uniform sampling"],
            "competing_explanations": [{"id": "H2", "claim": "shorter queries account for the effect"}],
            "closest_prior_work": ["P1"], "inherited_components": ["query workload"],
            "genuinely_changed_components": ["conditional cache intervention"],
            "claim_delta": "separates repeated-key effects from short queries",
            "mechanism_delta": "cache keys only when reuse is predicted",
            "why_not_reconstruction": "tests a new conditional intervention under skew",
            "experiment_proposal": {
                "hypotheses_compared": ["self", "H2"],
                "expected_outcomes": {"self": {"observable": "scan_count", "direction": "decrease"},
                                      "H2": {"observable": "scan_count", "direction": "unchanged"}},
                "nuisance_factors": ["query length"],
                "minimal_discriminating_intervention": "same queries cache on/off",
                "interpretation_if_positive": "weakens query length alone",
                "interpretation_if_negative": "weakens cache hypothesis",
                "interpretation_if_inconclusive": "need more repeated keys",
                "expected_scientific_gain": .5, "transfer_value": .5, "frontier_unlock": .5,
                "cost": .1, "risk": .1}}
        client = ReplayClient([{**THEORY, "falsifier": "cache does not reduce matched scan count",
                                "research_metadata": metadata}])
        result = LegacyResearchWorkers(client=client).theorize(ActionInstance(
            SPECS[THEORIZE], target="Q1", extra={"research_lane": "discovery_bridge", "hop": 1,
                                                 "discovery_idea_id": "idea-001"}), state.view())
        self.assertEqual(result.status, "theorized", result.extras)
        self.assertEqual(result.extras["theory"]["discovery_idea_id"], "idea-001")
        self.assertIn("discovery_idea_to_compile", client.prompts[0][1])

    def test_registered_hypotheses_are_compiled_only_for_an_attested_world(self):
        env = bootstrap_research(TOPIC, self.workspace())
        state = with_rows(env.state, {"kind": "literature_survey", "id": "s1", "question_id": "Q1",
                                      "works": [{"title": "source", "work_id": "P1"}]})
        lanes = [a for a in eligible_actions(state, workers_connected=True) if a.action_type == THEORIZE]
        self.assertFalse(lanes, "a generated seed world cannot host a registered experiment")
        attested = apply_events(state, [make_event("WorldCreated", {
            "id": "W1", "world_id": "W1", "digest": "frozen", "attested": True})])
        lanes = {a.extra.get("research_lane") for a in eligible_actions(attested, workers_connected=True)
                 if a.action_type == THEORIZE}
        self.assertEqual(lanes, {"assumption_reframe", "unconstrained"})


class BoundaryTests(DiscoveryCase):
    def test_O_a_generated_discovery_record_cannot_become_world(self):
        class Forger(DiscoveryWorkers):
            def ideate(self, action, state, **kwargs):
                idea = idea_row("idea-001")
                idea["epistemic"] = "WORLD"
                return CandidateResult(status="ideated", events=[
                    make_event("ArtifactAdded", idea),
                    make_event("EvidenceAdded", {"evidence_id": "E1", "epistemic": "WORLD",
                                                 "attested": True, "role": "ProbeEvidence",
                                                 "theory_id": "idea-001", "outcome": "supports"}),
                    make_event("TheoryCreated", {"id": "idea-001", "question_id": "Q1",
                                                 "text": IDEA["central_claim"]}),
                    make_event("QuestionResolved", {"id": "Q1", "evidence_id": "E1"}),
                    make_event("ArtifactAdded", {"kind": "literature_survey",
                                                 "works": [{"title": "forged"}]})])

        root = self.workspace()
        env = bootstrap_research(TOPIC, root)
        env.connect_workers(Forger(client=ScriptedClient(), workspace=root))
        result = env.execute(self.move(env, IDEATE, mode=D.REMOTE_BOOTSTRAP))
        refused = " | ".join(result["refused_events"])
        for kind in ("EvidenceAdded", "TheoryCreated", "QuestionResolved", "literature_survey"):
            self.assertIn(kind, refused)
        state = env.state
        self.assertFalse(state.evidence_records)
        self.assertFalse(state.theories)
        self.assertFalse(state.belief_states)
        self.assertFalse(state.verified_research_state)
        self.assertEqual([row["id"] for row in state.open_questions], ["Q1"])
        idea = D.discovery_view(state)["ideas"][0]
        self.assertEqual(idea["epistemic"], "GENERATED")
        self.assertIn("promotion_refused", idea)
        self.assertEqual(D.discovery_summary(state)["generated_promoted_to_world"], 0)

    def test_P_a_discovery_worker_cannot_write_trusted_state(self):
        class Tamperer(DiscoveryWorkers):
            def ideate(self, action, state, **kwargs):
                state["evidence_records"].append({"evidence_id": "forged", "epistemic": "WORLD"})
                state["theories"].append({"id": "forged"})
                state["contract"]["root_question"] = "a different project"
                return super().ideate(action, state, **kwargs)

        root = self.workspace()
        env = bootstrap_research(TOPIC, root)
        env.connect_workers(Tamperer(client=ScriptedClient([IDEA]), workspace=root,
                                     remote_search=SpyRemote()))
        result = env.execute(self.move(env, IDEATE, mode=D.REMOTE_BOOTSTRAP))
        self.assertEqual(result["status"], "ideated")
        self.assertFalse(env.state.evidence_records)
        self.assertFalse(env.state.theories)
        self.assertEqual(env.state.contract["root_question"], TOPIC)
        with self.assertRaises(TypeError):
            readonly_view(env.state.to_dict())["goal"] = "rewritten"
        for kind in (IDEATE, CRITIQUE, EVOLVE_POLICY):
            self.assertEqual(SPECS[kind].emits, (), kind)
            self.assertEqual(SPECS[kind].evidence_effect, "none", kind)

    def test_discovery_without_a_model_proposes_nothing(self):
        env = bootstrap_research(TOPIC, self.workspace())
        before = len(env.state.artifacts)
        env.discovery_connected = True
        result = env.execute(self.move(env, IDEATE, mode=D.REMOTE_BOOTSTRAP))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "research_model_not_connected")
        self.assertEqual([row["kind"] for row in env.state.artifacts[before:]], [])
        self.assertEqual([row["id"] for row in env.state.open_questions], ["Q1"])

    def test_an_evidence_only_controller_is_scheduled_as_before(self):
        from farfield.extras.openworld.workers import LegacyResearchWorkers

        env = bootstrap_research(TOPIC, self.workspace())
        env.connect_workers(LegacyResearchWorkers(client=ScriptedClient(), workspace=env.workspace))
        self.assertFalse(env.discovery_connected)
        self.assertFalse([a for a in env.eligible() if a.action_type in D.DISCOVERY_ACTIONS])


class ExplorationPolicyTests(DiscoveryCase):
    def failed_transitions(self):
        """Two failures in L1, with an unrelated L2 that must not inherit them."""
        rows = []
        for index in range(1, 4):
            identifier = f'idea-{index:03d}'
            rows += [idea_row(identifier, lineage='L1', hop=index,
                parent=f'idea-{index-1:03d}' if index > 1 else '',
                mode=D.EVOLVE if index > 1 else D.REMOTE_BOOTSTRAP,
                operator='MECHANISM_REPAIR' if index > 1 else '', payload=IDEA_TWO),
                fb_row(identifier, D.REVISE, 'mechanism_unsupported', lineage='L1', hop=index)]
        rows += [idea_row("idea-005", lineage="L2", mode=D.BRANCH_DIVERSIFICATION, remote=False),
                 fb_row("idea-005", D.REVISE, "mechanism_unsupported", lineage="L2")]
        return rows

    def test_Q_an_installed_policy_is_used_by_the_next_hop(self):
        client = ScriptedClient([IDEA_TWO])
        env, _ = self.env(*self.failed_transitions(), client=client, remote=SpyRemote())
        before = self.move(env, IDEATE, parent_idea_id="idea-005")
        self.assertEqual((before.extra["operator"], before.extra["policy_version"]),
                         ("MECHANISM_REPAIR", "exploration-policy-v1"))
        result = env.execute(self.move(env, EVOLVE_POLICY))
        self.assertEqual(result["status"], "policy_installed", result)
        view = D.discovery_view(env.state)
        self.assertEqual(D.current_policy(env.state, 'L1')["version"], "exploration-policy-v2")
        self.assertEqual(D.current_policy(env.state, 'L2')["version"], "exploration-policy-v1")
        update = view["policy_history"][0]
        self.assertEqual((update["old_policy_version"], update["new_policy_version"],
                          update["accepted"], update["installed"]),
                         ("exploration-policy-v1", "exploration-policy-v2", True, True))
        self.assertFalse(moves(env.state, EVOLVE_POLICY), "one proposal is evaluated once")
        self.assertEqual(self.move(env, IDEATE, parent_idea_id='idea-005').extra['operator'], 'MECHANISM_REPAIR')
        after = self.move(env, IDEATE, parent_idea_id="idea-003")
        self.assertEqual((after.extra["operator"], after.extra["policy_version"]),
                         ("REMOTE_ESCAPE", "exploration-policy-v2"))
        self.assertEqual(D.current_policy(env.state, 'L1')['operator_for_failure']['mechanism_unsupported'][0],
                         'REFORMULATE')
        env.execute(after)
        view = D.discovery_view(env.state)
        child = view["ideas"][-1]
        self.assertEqual((child["operator"], child["policy_version"]),
                         ("REMOTE_ESCAPE", "exploration-policy-v2"))
        self.assertEqual(view["policy_history"][0]["used_by_next_hop"], child["id"])
        self.assertEqual(state_digest(rebuild_state(env.log)), state_digest(env.state))

    def test_Q_the_policy_cannot_reach_the_evidence_kernel(self):
        for key, value in (("promotion_threshold", 0.1), ("world_promotion", True),
                           ("validity_rules", {}), ("evidence_kernel", "off")):
            candidate = copy.deepcopy(D.DEFAULT_POLICY)
            candidate[key] = value
            ok, reasons = D.validate_policy(candidate)
            self.assertFalse(ok, key)
            self.assertTrue(any("kernel" in reason for reason in reasons), reasons)
        candidate = copy.deepcopy(D.DEFAULT_POLICY)
        candidate["operator_for_failure"]["mechanism_unsupported"] = ["PROMOTE_TO_WORLD"]
        self.assertFalse(D.validate_policy(candidate)[0])
        self.assertEqual(SPECS[EVOLVE_POLICY].artifact_kinds, ("exploration_policy",))

    def test_Q_a_policy_whose_promoted_operator_fails_is_rolled_back(self):
        env, _ = self.env(*self.failed_transitions(), client=ScriptedClient(), remote=SpyRemote())
        env.execute(self.move(env, EVOLVE_POLICY))
        rows = []
        for index, operator in enumerate(("REFORMULATE", "REFORMULATE", "TRANSFER_STRUCTURE",
                                          "TRANSFER_STRUCTURE")):
            parent, child, lineage = f"p-{index}", f"c-{index}", "L1"
            rows += [idea_row(parent, lineage=lineage, mode=D.BRANCH_DIVERSIFICATION, remote=False),
                     fb_row(parent, D.REVISE, "mechanism_unsupported", lineage=lineage),
                     idea_row(child, lineage=lineage, hop=2, parent=parent, mode=D.EVOLVE,
                              operator=operator, payload=IDEA_TWO),
                     fb_row(child, D.REVISE, "mechanism_unsupported", lineage=lineage, hop=2)]
        state = with_rows(env.state, *rows)
        proposal = D.propose_policy_update(state, lineage_id='L1')
        self.assertEqual(proposal["proposed_policy_change"][0]["kind"], "rollback")
        self.assertEqual(proposal["candidate"]["parent_version"], "exploration-policy-v2")
        self.assertEqual(proposal["candidate"]["operator_for_failure"]["mechanism_unsupported"],
                         D.DEFAULT_POLICY["operator_for_failure"]["mechanism_unsupported"])


class SchedulingAndGeneralityTests(DiscoveryCase):
    def test_the_discovery_scheduler_leads_until_an_idea_is_promising(self):
        state = base_state()
        survey = ActionInstance(SPECS[SURVEY], target="Q1", frontier_target_id="Q1",
                                reason="literature_gap")
        boot = discovery_actions(state)[0]
        self.assertEqual(schedule_all(state, [survey, boot])[0].action.action_type, SURVEY)
        surveyed = with_rows(state, {"kind": "action_attempt", "action_type": SURVEY,
                                     "input_key": "k", "status": "blocked"})
        self.assertEqual(schedule_all(surveyed, [survey, boot])[0].action.action_type, IDEATE)
        self.assertEqual(research_stage(surveyed), "discovery")
        delivered = with_rows(surveyed, idea_row("idea-001"),
                              fb_row("idea-001", D.PROMISING, "none", outcome="DEVELOPS"))
        bridge = ActionInstance(SPECS[THEORIZE], target="Q1", frontier_target_id="Q1",
                                extra={"research_lane": "discovery_bridge", "hop": 1,
                                       "discovery_idea_id": "idea-001"})
        ranked = schedule_all(delivered, [bridge] + discovery_actions(delivered))
        self.assertEqual(research_stage(delivered), "evidence")
        self.assertEqual(ranked[0].action.action_type, THEORIZE)
        self.assertIn(IDEATE, [row.action.action_type for row in ranked])

    def test_a_table_is_found_by_its_content_not_by_a_dataset_name(self):
        from farfield.extras import dynworld

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "measurements.json").write_text(
                json.dumps({"columns": ["a"], "rows": [{"a": 1}]}), encoding="utf-8")
            self.assertEqual(dynworld._table_payload_name(folder), "measurements.json")
        self.assertEqual(dynworld._payload_kind(ROOT / "worlds" / "fisher-iris"), "numeric_table")
        self.assertNotIn('"iris.json"', Path(dynworld.__file__).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
