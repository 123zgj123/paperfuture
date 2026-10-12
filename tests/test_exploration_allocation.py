"""Budget and isolation contracts, not tests of scientific idea quality."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from farfield.extras.llm import TokenLedger, LLMUnavailable
from farfield.extras.openworld import discovery as D
from farfield.extras.openworld.actions import ActionInstance, SPECS
from farfield.extras.openworld.ideation import critique, ideate, _literature
from farfield.extras.openworld.memory import lineage_learning
from farfield.extras.openworld.resources import allocate_runnable, action_call_estimate
from farfield.extras.openworld.state import ScientificState
from test_discovery_runtime import IDEA, ScriptedClient, base_state, idea_row, fb_row, with_rows


class AllocationTests(unittest.TestCase):
    def state(self, remaining=3, limits=None):
        return ScientificState(goal='research', contract={'root_question': 'research',
            'research_mode': 'open_research', 'resource_limits': limits or {}},
            resources={'rule': 'measured-spend-v1', 'remaining': {'api_calls': remaining},
                       'limits': {'api_calls': 20}, 'stage_spent': {'discovery': 12}})

    def test_opening_survey_cannot_spend_the_next_idea_and_critique(self):
        survey = ActionInstance(SPECS['SURVEY'])
        idea = ActionInstance(SPECS['IDEATE'])
        self.assertEqual(allocate_runnable(self.state(), [survey, idea])[0], [idea])
        self.assertEqual(allocate_runnable(self.state(8), [survey, idea])[0], [survey, idea])

    def test_default_can_continue_exploration_past_old_share(self):
        actions = [ActionInstance(SPECS['IDEATE']), ActionInstance(SPECS['SURVEY'])]
        self.assertEqual(allocate_runnable(self.state(8), actions)[0], actions)
        self.assertEqual(allocate_runnable(self.state(8, {'discovery_share': .4}), actions)[0], actions[1:])

    def test_targeted_literature_and_only_available_route_remain_possible(self):
        survey = ActionInstance(SPECS['SURVEY'], extra={'theory_id': 'I'})
        idea = ActionInstance(SPECS['IDEATE'])
        self.assertEqual(allocate_runnable(self.state(), [survey, idea])[0], [survey, idea])
        self.assertEqual(allocate_runnable(self.state(), [survey])[0], [survey])

    def test_fractional_calls_cannot_start_a_whole_request_in_either_client(self):
        from farfield.services.llm import TokenLedger as ServiceLedger, LLMUnavailable as ServiceUnavailable
        for cls, error in ((TokenLedger, LLMUnavailable), (ServiceLedger, ServiceUnavailable)):
            ledger = cls(api_calls=.5, token_cost=100)
            with self.assertRaises(error):
                ledger.reserve_call('test')
            self.assertEqual(ledger.spent_calls, 0)
            self.assertEqual(ledger.reserved_calls, 0)

    def test_last_call_completes_joint_feedback_without_starting_partial_rollout(self):
        client = ScriptedClient()
        client.ledger = TokenLedger(api_calls=1, token_cost=10000)
        state = with_rows(base_state(), idea_row('I'))
        action = ActionInstance(SPECS['CRITIQUE'], target='I', extra={'idea_id': 'I'})
        with patch.dict('os.environ', {}, clear=True):
            result = critique(action, state, client=client, workspace=None)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], 'discovery_critique')
        self.assertFalse(result.events[0].payload['rollout_separate_from_critique'])
        self.assertFalse(result.events[0].payload['budget_execution']['quality_equivalence_verified'])

    def test_critique_cost_matches_adaptive_mode(self):
        action = ActionInstance(SPECS['CRITIQUE'])
        self.assertEqual(action_call_estimate(self.state(1), action), 1)
        self.assertEqual(action_call_estimate(self.state(12), action), 2)


class IsolationTests(unittest.TestCase):
    def state(self):
        return with_rows(base_state(), idea_row('A', lineage='L1'), idea_row('B', lineage='L2'),
            {**fb_row('A', D.REVISE, 'measurement_invalid'), 'recommended_next_operator': 'MEASUREMENT_REPAIR',
             'mistakes_to_avoid': ['A-only constraint']},
            {**fb_row('B', D.REVISE, 'scope_mismatch', lineage='L2'),
             'recommended_next_operator': 'SCOPE_SHIFT', 'mistakes_to_avoid': ['B-only constraint']})

    def test_only_selected_ancestry_enters_memory_and_procedures(self):
        state = self.state().to_dict()
        packet = lineage_learning(state, lineage_id='L1', parent_id='A')
        self.assertEqual([r['idea_id'] for r in packet['memory']], ['A'])
        self.assertNotIn('B-only constraint', json.dumps(packet))
        self.assertFalse(packet['shared_installation'])
        self.assertFalse(packet['skill_candidates'][0]['executable'])
        packet['memory'][0]['critic_feedback']['mistakes_to_avoid'].append('mutated')
        self.assertNotIn('mutated', json.dumps(state))

    def test_new_task_inherits_no_memory_or_policy(self):
        packet = lineage_learning(base_state().to_dict(), lineage_id='L1', parent_id='A')
        self.assertEqual(packet['memory'], [])
        self.assertEqual(packet['skill_candidates'], [])
        self.assertEqual(D.current_policy(base_state(), 'L1'), D.DEFAULT_POLICY)

    def test_project_pivot_cannot_reuse_a_previous_epoch_policy_or_memory(self):
        policy = copy.deepcopy(D.DEFAULT_POLICY)
        policy['version'] = 'old-task-policy'
        state = with_rows(self.state(), {'kind': D.POLICY_KIND, 'id': 'p', 'installed': True,
            'lineage_id': 'L1', 'policy': policy})
        state.project_epoch = 1
        self.assertEqual(D.current_policy(state, 'L1'), D.DEFAULT_POLICY)
        self.assertEqual(D.policy_records(state), [])
        self.assertEqual(lineage_learning(state.to_dict(), lineage_id='L1', parent_id='A')['memory'], [])

    def test_ambiguous_legacy_policy_cannot_enter_lineage_rollback_history(self):
        state = with_rows(self.state(), {'kind': D.POLICY_KIND, 'id': 'legacy',
            'installed': True, 'policy': copy.deepcopy(D.DEFAULT_POLICY)})
        self.assertEqual(D.scoped_policy_records(state, 'L1'), [])
        self.assertEqual(D.scoped_policy_records(state, 'L2'), [])

    def test_policy_in_one_lineage_cannot_change_another(self):
        policy = copy.deepcopy(D.DEFAULT_POLICY)
        policy['version'] = 'local-v2'
        state = with_rows(self.state(), {'kind': D.POLICY_KIND, 'id': 'p', 'installed': True,
            'lineage_id': 'L1', 'policy': policy})
        self.assertEqual(D.current_policy(state, 'L1')['version'], 'local-v2')
        self.assertEqual(D.current_policy(state, 'L2'), D.DEFAULT_POLICY)
        self.assertEqual(D.current_policy(state), D.DEFAULT_POLICY)

    def test_actual_ideation_payload_uses_local_learning_not_sibling_mistakes(self):
        client = ScriptedClient([IDEA])
        state = self.state()
        action = ActionInstance(SPECS['IDEATE'], target='A', extra={'mode': D.EVOLVE,
            'lineage_id': 'L1', 'parent_idea_id': 'A', 'operator': 'MEASUREMENT_REPAIR'})
        result = ideate(action, state, client=client, workspace=None)
        self.assertEqual(result.status, 'ideated')
        prompt = json.loads(client.calls[0][1].split('\n\n', 1)[1])
        self.assertNotIn('B-only constraint', json.dumps(prompt['current_research_state']['local_learning']))
        self.assertNotIn('B-only constraint', json.dumps(prompt['archive_mistakes_to_avoid']))

    def test_off_topic_and_pending_papers_are_not_presented_as_grounding(self):
        state = {'artifacts': [{'kind': 'literature_survey', 'works': [
            {'title': 'A', 'arxiv_id': '2501.00001', 'published': '2025-01-01'},
            {'title': 'B', 'arxiv_id': '2501.00002', 'published': '2025-01-02'}],
            'verification': [{'cite_id': '2501.00001', 'status': 'verified'},
                             {'cite_id': '2501.00002', 'status': 'verify_pending'}],
            'retrieval_relevance': {'usable': ['2501.00002']}}]}
        self.assertEqual(_literature(state)['works'], [])


if __name__ == '__main__':
    unittest.main()
