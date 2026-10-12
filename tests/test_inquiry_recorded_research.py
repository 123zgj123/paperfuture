"""Engineering regressions replaying recorded real research, not generated worlds.

Optional public/audit assets remain outside git. These checks do not measure
scientific benefit; scripts/eval/scifact_sources.py uses the official scorer.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from farfield.extras.openworld.actions import ActionInstance, SPECS
from farfield.extras.openworld.discovery import discovery_records
from farfield.extras.openworld.events import make_event
from farfield.extras.openworld.ideation import _literature, ideation_prompt
from farfield.extras.openworld.inquiry import (
    investment_packet, inquiry_request, inquiry_progress, inquiry_is_due, knowledge_move,
)
from farfield.extras.openworld.reducer import apply_events, state_digest
from farfield.extras.openworld.scheduler import schedule_all
from farfield.extras.openworld.sourcecheck import check_sources
from farfield.extras.openworld.sourcecheck import reading_sources
from farfield.extras.openworld.understanding import technical_map
from farfield.extras.openworld.state import ScientificState


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'docs/audits/2026-10-08-system-review/exploration-budget-live/aib-cs/research/SCIENTIFIC_STATE.json'
PILOT = ROOT/'docs/audits/2026-10-09-opportunity/scifact-pilot'
BIOLOGY = ROOT/'docs/audits/2026-10-08-system-review/external-benchmarks/aib-biology/research/SCIENTIFIC_STATE.json'


@unittest.skipUnless(BIOLOGY.is_file(), 'requires preserved real full-text source documents')
class RecordedFullTextRouting(unittest.TestCase):
    def setUp(self):
        self.state = ScientificState.from_dict(json.loads(BIOLOGY.read_text()))
        self.survey = next(r for r in self.state.artifacts if r.get('source_documents'))

    def test_saved_excerpts_are_bounded_and_wrong_document_identity_cannot_supply_text(self):
        import hashlib
        docs = self.survey['source_documents']
        material = reading_sources(self.survey['works'], docs, max_characters=24000)
        self.assertLessEqual(sum(len(s['text']) for s in material), 24000)
        full = [s for s in material if s['coverage'] == 'full_text_excerpt']
        self.assertEqual(len(full), 2)
        for s in full:
            self.assertTrue(s['truncated'])
            self.assertEqual(s['content_sha256'], hashlib.sha256(s['text'].encode()).hexdigest())
            original = docs[s['source_id']]['full_text_spans'][0]
            self.assertTrue(original['text'].startswith(s['text']))
            self.assertEqual(s['source_spans'][0]['source_digest'], original['source_digest'])
        wrong = copy.deepcopy(docs)
        for doc in wrong.values():
            doc['arxiv_id'] = 'unbound-control-id'
        self.assertTrue(all(s['coverage'] == 'abstract_only'
                            for s in reading_sources(self.survey['works'], wrong)))

    def test_both_production_readers_receive_the_saved_full_text_without_evidence_promotion(self):
        import tempfile
        from farfield.extras.livefeed import FreshWork
        from farfield.extras.openworld.workers import LegacyResearchWorkers
        docs, works, captured = self.survey['source_documents'], self.survey['works'], {}
        ledger = SimpleNamespace(remaining=lambda: {'api_calls': 100})
        class Feed:
            verify_literature = False  # Replay already recorded metadata without a network request.
            def survey_around(self, *args, **kwargs):
                return [FreshWork.from_dict(w) for w in works]
            def read_work(self, work):
                return docs.get(work.cite_id(), {})
        class Client:
            def __init__(self):
                self.ledger = ledger
            def complete(self, prompt, *, purpose, **kwargs):
                if purpose == 'retrieval_search_topic':
                    body = {'topic': 'bacteriophage therapy', 'subject': 'bacteriophage therapy'}
                else:
                    payload = json.loads(prompt)
                    captured[purpose] = payload
                    # This stub measures only input delivery and admission.
                    body = {'assessments': [{'source_id': s['source_id'],
                            'label': 'NOT_ENOUGH_INFO', 'sentences': []} for s in payload['sources']]}
                return SimpleNamespace(text=json.dumps(body))
        with tempfile.TemporaryDirectory() as tmp:
            worker = LegacyResearchWorkers(feed=Feed(), workspace=Path(tmp),
                                           client=Client(), knowledge_client=Client())
            result = worker.survey(ActionInstance(SPECS['SURVEY'], target='routing-check',
                extra={'knowledge_request': {'question': self.state.goal}}), self.state)
        self.assertEqual(result.status, 'surveyed')
        artifact = result.events[0].payload
        revised = {s['source_id']: s for s in captured['research_technical_map_revise']['sources']}
        checked = {s['source_id']: s for s in captured['research_source_check']['sources']}
        for s in artifact['reading_sources']:
            self.assertEqual(revised[s['source_id']]['text'], s['text'])
            self.assertEqual(checked[s['source_id']]['sentences'], s['sentences'])
        self.assertEqual(sum(s['coverage'] == 'full_text_excerpt' for s in artifact['reading_sources']), 2)
        self.assertFalse(result.evidence)
        self.assertFalse(any(e.event_type == 'EvidenceAdded' for e in result.events))


@unittest.skipUnless(OLD.is_file(), 'requires preserved real AgentIdeaBench-topic run')
class RecordedResearch(unittest.TestCase):
    def setUp(self):
        self.state = ScientificState.from_dict(json.loads(OLD.read_text()))
        ideas, feedback = discovery_records(self.state)
        self.idea, self.feedback = ideas[-1], feedback[ideas[-1]['id']]
        # Control-plane budget variation, retaining the exact scientific input.
        self.state.resources['remaining']['api_calls'] = 20
        self.state.resources['remaining']['ticks'] = 10
        self.state.resources['exhausted'] = []
        self.state.resources['limits'].update(api_calls=24, ticks=13)
        self.feedback['investment'] = investment_packet(self.idea['idea'], self.feedback,
            {'status': 'literature_gap'}, self.state)
        for row in self.state.artifacts:
            if row.get('id') == self.feedback['id']:
                row['investment'] = copy.deepcopy(self.feedback['investment'])
        self.request = inquiry_request(self.idea, self.feedback)

    def test_real_idea_gets_a_bounded_schedulable_inquiry(self):
        before = state_digest(self.state)
        move = knowledge_move(self.state, self.idea, self.feedback)
        self.assertIsNotNone(move)
        action = ActionInstance(SPECS[move['action']], target=move['target'],
                                frontier_target_id='goal:G0', extra=move['extra'])
        self.assertTrue(schedule_all(self.state, [action]))
        self.assertEqual(state_digest(self.state), before)

    def test_low_budget_retains_the_idea_as_under_assessed(self):
        self.state.resources['remaining']['api_calls'] = 2
        self.assertIsNone(knowledge_move(self.state, self.idea, self.feedback))
        result = inquiry_progress(self.state, self.request)
        self.assertEqual(result['status'], 'budget_insufficient')
        self.assertEqual(result['assessment'], 'under_assessed_not_refuted')

    def test_attempted_survey_is_not_rebought_after_replay(self):
        after = apply_events(self.state, [make_event('ArtifactAdded', {
            'kind': 'action_attempt', 'action_type': 'SURVEY', 'target': self.request['id'],
            'project_epoch': self.state.project_epoch, 'inquiry_lineage_id': self.request['lineage_id']})])
        restored = ScientificState.from_dict(after.to_dict())
        self.assertFalse(inquiry_is_due(restored, self.request))
        self.assertEqual(inquiry_progress(restored, self.request)['status'], 'attempted_without_sources')

    def test_action_cannot_replace_the_recorded_inquiry_question(self):
        changed = dict(self.request, question=self.idea['idea']['central_claim'])
        if changed == self.request:
            changed['question'] = self.idea['idea']['mechanism']
        self.assertFalse(inquiry_is_due(self.state, changed))

    def test_current_literature_projections_preserve_recorded_source_text(self):
        context = _literature(self.state.to_dict())
        for passage in context['source_passages']:
            self.assertTrue(any(passage['text'] in s.get('text', '')
                for r in self.state.artifacts if r.get('kind') == 'literature_survey'
                for s in r.get('spans', [])))
        prompt = ideation_prompt(mode='REMOTE_BOOTSTRAP', root=self.state.goal, literature=context)
        self.assertIn('source_passages', prompt)


@unittest.skipUnless((PILOT/'retrieval.json').is_file(), 'requires real SciFact pilot records')
class RecordedSourceCalls(unittest.TestCase):
    def test_frontier_maps_replay_without_promoting_missing_chronology(self):
        completed = sorted(PILOT.glob('*-knowledge/revision.json'))
        if not completed:
            self.skipTest('no completed public technical-map calls')
        for saved_path in completed:
            folder = saved_path.parent
            for phase, name in [('sketch', 'sketch.json'), ('revise', 'revision.json')]:
                with self.subTest(folder=folder.name, phase=phase):
                    request_path = next(p for p in folder.glob('*-request.json')
                        if json.loads(p.read_text())['purpose'] == 'research_technical_map_' + phase)
                    request = json.loads(request_path.read_text())
                    response = json.loads(request_path.with_name(
                        request_path.name.replace('-request', '-response')).read_text())
                    payload = json.loads(request['body']['messages'][1]['content'])
                    class Replay:
                        def complete(self, *args, **kwargs):
                            return SimpleNamespace(text=response['choices'][0]['message']['content'],
                                model=response.get('model', ''), digest='')
                    result = technical_map(Replay(), payload['research_question'],
                        sources=payload['sources'], previous=payload['previous_map'])
                    saved = json.loads((folder/name).read_text())
                    for field in ('transitions', 'revisions', 'reference_gaps', 'source_ids'):
                        self.assertEqual(result[field], saved[field])
                    self.assertFalse(result['scientific_validation'])
                    self.assertEqual(result['epistemic'], 'GENERATED')
                    self.assertTrue(all(t['status'] == 'unverified_hypothesis'
                                        for t in result['transitions']))

    def test_actual_model_responses_replay_through_production_validator(self):
        folders = sorted(PILOT.glob('*-sourcecheck'))
        completed = [p for p in folders if (p/'interpreted.json').is_file()]
        if not completed:
            self.skipTest('no completed public source-check calls')
        for folder in completed:
            with self.subTest(folder=folder.name):
                request = json.loads(next(folder.glob('*-request.json')).read_text())
                response = json.loads(next(folder.glob('*-response.json')).read_text())
                payload = json.loads(request['body']['messages'][1]['content'])
                class Replay:
                    def complete(self, *args, **kwargs):
                        return SimpleNamespace(text=response['choices'][0]['message']['content'],
                                               model=response.get('model', ''), digest='')
                output = check_sources(Replay(), payload['claim'], payload['sources'])
                saved = json.loads((folder/'interpreted.json').read_text())
                self.assertEqual(output['assessments'], saved['assessments'])
                self.assertEqual(output['epistemic'], 'GENERATED')


if __name__ == '__main__':
    unittest.main()
