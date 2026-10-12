"""Regression tests for the substantive v0.3 failure modes, not model quality."""
import copy
import datetime as dt
import hashlib
import io
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import unittest
import test_paperfuture as base
from paperfuture.config import normalize
from paperfuture.contracts import ContractError
from paperfuture.dataset import Archive, discover_window
from paperfuture.exploration import experience_prompt, score_assessment
from paperfuture.graph import graph_context
from paperfuture.literature import Snapshot, evidence_view, paper_record, round_context, temporal_rank
from paperfuture.storage import read_json,unseal
from paperfuture_support import SyntheticEndpoint,papers


class PaperFutureV03Tests(unittest.TestCase):
    setUp=base.PaperFutureTests.setUp
    execute=base.PaperFutureTests.execute
    directory=base.PaperFutureTests.directory

    def test_parallel_graph_calls_keep_request_identities_separate(self):
        class Concurrent(SyntheticEndpoint):
            lock=threading.Lock()
            active=0
            maximum=0
            def respond(self,role,data):
                if role!='graph_edges':return super().respond(role,data)
                with self.lock:
                    self.active+=1;self.maximum=max(self.maximum,self.active)
                try:
                    time.sleep(.03)
                    return super().respond(role,data)
                finally:
                    with self.lock:self.active-=1
        endpoint=Concurrent();self.execute(endpoint)
        self.assertGreaterEqual(endpoint.maximum,2)
        identities=[]
        for path in (self.directory()/'api/receipts').glob('*.json'):
            request=unseal(path)['request']
            if request['role']=='graph_edges':identities.append(request['identity'])
        self.assertEqual(len(identities),len(set(identities)))

    def test_concurrent_graph_workers_share_one_atomic_api_budget(self):
        config=copy.deepcopy(self.config);config['api']['max_calls']=9
        with self.assertRaises(base.BudgetExceeded):self.execute(config=config)
        result=read_json(self.directory()/'final_idea.json')
        self.assertLessEqual(result['usage']['network_attempts'],9)
        self.assertEqual(result['status'],'interrupted')

    def test_preparation_reuse_preserves_receipts_but_never_reuses_new_candidates(self):
        self.execute()
        second=copy.deepcopy(self.config);second['run_id']='second-run'
        result,endpoint=self.execute(config=second)
        self.assertEqual([role for role,_ in endpoint.requests],['propose','simulate','critique','discuss','assess'])
        reuse=result['usage']['preparation_cache_reuses']
        self.assertEqual(len(reuse),11)
        self.assertTrue(all(Path(r['provenance']['source_receipt']).exists() for r in reuse))
        third=copy.deepcopy(self.config);third.update(run_id='independent-replicate',replicate_id='2')
        _,endpoint=self.execute(config=third)
        self.assertIn('graph_stages',[role for role,_ in endpoint.requests])

    def test_keyboard_interruption_is_recorded_and_resumes_same_candidate(self):
        class Interrupted(SyntheticEndpoint):
            interrupted=False
            def respond(self,role,data):
                if role=='critique' and not self.interrupted:
                    self.interrupted=True
                    raise KeyboardInterrupt()
                return super().respond(role,data)
        endpoint=Interrupted()
        with self.assertRaises(KeyboardInterrupt): self.execute(endpoint)
        state=unseal(self.directory()/'state.json')
        self.assertEqual(state['status'],'interrupted')
        self.assertEqual(state['error'],'KeyboardInterrupt')
        result,_=self.execute(endpoint)
        self.assertEqual(result['reserved_rounds'],1)
        self.assertEqual(sum(role=='propose' for role,_ in endpoint.requests),1)

    def test_short_evidence_aliases_round_trip_to_exact_archived_spans(self):
        _, endpoint = self.execute()
        extraction = next(data for role, data in endpoint.requests if role == 'extract')
        self.assertTrue(all(k.startswith('E') and k[1:].isdigit() for k in extraction['evidence_spans']))
        context = unseal(self.directory()/'rounds/01/evidence_context.json')
        self.assertTrue(all(k.startswith('ev_') for k in context['evidence']))
        proposal = unseal(self.directory()/'rounds/01/proposal.json')
        self.assertTrue(set(proposal['evidence_ids']) <= set(context['evidence']))
        self.assertTrue(list((self.directory()/'api/evidence_aliases').glob('*.json')))

    def test_unknown_short_evidence_alias_is_never_fuzzily_repaired(self):
        class Inventing(SyntheticEndpoint):
            def respond(self,role,data):
                result=super().respond(role,data)
                if role=='extract': result['evidence_ids']=['E999999']
                return result
        with self.assertRaisesRegex(ContractError,'unknown references'):
            self.execute(Inventing())

    def test_failed_source_window_is_split_without_hiding_failure_or_double_counting(self):
        class Source:
            def __init__(self): self.calls=[]
            def query(self,params):
                self.calls.append(params)
                if len(self.calls)==1: raise OSError('synthetic oversized source query')
                return b'<feed xmlns:o="http://a9.com/-/spec/opensearch/1.1/"><o:totalResults>1</o:totalResults></feed>', Path('fixture.xml')
        source=Source();seen=[]
        def record(raw,path): seen.append(path);return [path]
        result=discover_window(source,'fixture',dt.datetime(2026,1,1,tzinfo=dt.timezone.utc),
                               dt.datetime(2026,1,31,23,59,tzinfo=dt.timezone.utc),record)
        self.assertIsNone(result['error'])
        self.assertEqual(result['initial_error'],'OSError')
        self.assertEqual(result['retrieved'],result['matched'])
        self.assertEqual(result['retrieved'],2)
        left,right=result['children']
        self.assertEqual(dt.datetime.fromisoformat(left['end'])+dt.timedelta(minutes=1),dt.datetime.fromisoformat(right['start']))

    def test_stage_batches_are_reconciled_before_edges(self):
        config=copy.deepcopy(self.config);config['graph']['stage_batch']=2
        _,endpoint=self.execute(config=config)
        roles=[role for role,_ in endpoint.requests]
        self.assertEqual(roles.count('graph_stages'),2)
        self.assertLess(roles.index('graph_merge'),roles.index('graph_edges'))
        graph=unseal(self.directory()/'trajectory.json')
        self.assertEqual({k for s in graph['stages'] for k in s['paper_keys']},{p['paper_id']+'@v1' for p in papers()})

    def test_shared_topic_cannot_merge_different_technical_mechanisms(self):
        class Coarse(SyntheticEndpoint):
            def respond(self,role,data):
                result=super().respond(role,data)
                if role=='graph_merge':
                    groups=result['groups'];other=groups.pop()['representative_stage_id']
                    groups[0]['member_stage_ids'].append(other)
                    groups[0]['comparisons']=[{'stage_id':other,'same_problem':True,'same_mechanism':False,
                        'same_assumptions':True,'same_capability_boundary':True,'reason':'same broad topic, different mechanism'}]
                return result
        config=copy.deepcopy(self.config);config['graph']['stage_batch']=2
        with self.assertRaisesRegex(ContractError,'different technical regimes'):
            self.execute(Coarse(),config)

    def test_dimension_citations_are_aggregated_without_inventing_evidence(self):
        class DimensionReferences(SyntheticEndpoint):
            def respond(self,role,data):
                result=super().respond(role,data)
                if role=='graph_edges':
                    for edge in result['edges']: edge['evidence_ids']=[]
                return result
        self.execute(DimensionReferences())
        graph=unseal(self.directory()/'trajectory.json')
        for edge in graph['proposed_edges']:
            self.assertIn(edge['target_witness'],edge['evidence_ids'])
            self.assertTrue(all(set(reason['evidence_ids'])<=set(edge['evidence_ids']) for reason in edge['dimension_reasons'].values()))

    def test_evidenced_conflict_cannot_disappear_with_empty_resolution_references(self):
        result,_=self.execute();r=result['candidate']
        c=copy.deepcopy(r['critique']);a=copy.deepcopy(r['assessment'])
        c['objections'][0].update(kind='evidenced_conflict',severity=1,evidence_ids=['valid-original-evidence'])
        a['resolutions'][0]['evidence_ids']=[]
        scores=score_assessment(a,c,self.config['policy'])
        self.assertEqual(scores['critic_penalty'],1)
        self.assertEqual(scores['blocking_objection_ids'],['C1'])

    def test_integration_only_cannot_receive_strong_recommendation(self):
        class Stacking(SyntheticEndpoint):
            def respond(self,role,data):
                r=super().respond(role,data)
                if role=='assess':r['mechanism_audit']['kind']='integration_only'
                return r
        result,_=self.execute(Stacking())
        self.assertEqual(result['candidate']['scores']['effective_novelty'],2)
        self.assertNotEqual(result['recommendation'],'recommend_for_empirical_validation')
        self.assertEqual(result['completed_rounds'],4)

    def test_already_reported_mechanism_has_zero_novelty_despite_high_raw_score(self):
        class RepeatingPriorWork(SyntheticEndpoint):
            def respond(self,role,data):
                r=super().respond(role,data)
                if role=='assess':r['mechanism_audit']['kind']='already_known'
                return r
        result,_=self.execute(RepeatingPriorWork())
        self.assertEqual(result['candidate']['scores']['effective_novelty'],0)
        self.assertNotEqual(result['recommendation'],'recommend_for_empirical_validation')
        self.assertEqual(result['completed_rounds'],4)

    def test_independent_edge_review_changes_rating_and_drops_unsupported_edge(self):
        class Reviewing(SyntheticEndpoint):
            def respond(self,role,data):
                r=super().respond(role,data)
                if role=='graph_review':
                    for review in r['reviews']:
                        if review['edge_id']=='e1': review['verdict']='unsupported'
                return r
        self.execute(Reviewing())
        g=unseal(self.directory()/'trajectory.json')
        self.assertEqual(len(g['proposed_edges']),3)
        self.assertEqual(len(g['edges']),2)
        self.assertEqual(g['proposed_edges'][1]['weight'],65)
        self.assertEqual(g['edges'][0]['weight'],63)
        self.assertEqual(g['edges'][0]['delta']['mechanism'],7)

    def test_zero_supported_edges_cannot_be_presented_as_successful_trajectory(self):
        class Unsupported(SyntheticEndpoint):
            def respond(self,role,data):
                r=super().respond(role,data)
                if role=='graph_review':
                    for review in r['reviews']:review['verdict']='unsupported'
                return r
        with self.assertRaisesRegex(ValueError,'no development edges'):
            self.execute(Unsupported())
        self.assertEqual(read_json(self.directory()/'final_idea.json')['status'],'interrupted')

    def test_panel_uses_distinct_models_and_preserves_each_discussion(self):
        config=copy.deepcopy(self.config)
        config['models']['simulate_secondary']={**config['models']['default'],'model':'synthetic-second-model'}
        config['world_model'].update(simulators=['simulate','simulate_secondary'],min_distinct_models=2)
        result,endpoint=self.execute(config=config)
        self.assertEqual(len(result['candidate']['simulation']['members']),2)
        self.assertEqual(len(result['candidate']['discussion']['members']),2)
        self.assertEqual(sum(role=='simulate' for role,_ in endpoint.requests),2)
        self.assertEqual(len({m['model_id'] for m in result['candidate']['simulation']['members']}),2)
        config['models']['simulate_secondary']['model']=config['models']['default']['model']
        with self.assertRaisesRegex(ValueError,'distinct configured model'):
            normalize(config)

    def test_fourth_leap_has_all_prior_experiences_and_changes_retrieval_query(self):
        _,endpoint=self.execute(SyntheticEndpoint(high_value=False))
        fourth=[v for role,v in endpoint.requests if role=='propose'][3]
        self.assertEqual({e['round'] for e in fourth['cumulative_experiences']},{1,2,3})
        self.assertTrue(all(e['epistemic_status']!='historical_evidence' for e in fourth['cumulative_experiences']))
        first=unseal(self.directory()/'rounds/01/evidence_context.json')['round_retrieval']
        second=unseal(self.directory()/'rounds/02/evidence_context.json')['round_retrieval']
        self.assertNotEqual(first['query'],second['query'])
        self.assertTrue(second['experience_ids'])

    def test_experience_prompt_deduplicates_assumptions_without_losing_dependencies(self):
        assumptions=[{'id':f'a{i}','statement':'conditional hypothesis '*100,
                      'origin':'hypothesis','evidence_ids':[]} for i in range(12)]
        experiences=[{'id':f'lesson-{i}','round':i//10+1,'content':'conditional prediction',
                      'epistemic_status':'conditional_simulation_not_evidence',
                      'assumptions':copy.deepcopy(assumptions)} for i in range(30)]
        items,catalogue=experience_prompt(experiences)
        restored=[{k:v for k,v in item.items() if k!='assumption_ids'} |
                  {'assumptions':[catalogue[key] for key in item['assumption_ids']]} for item in items]
        self.assertEqual(restored,experiences)
        self.assertLess(len(json.dumps([items,catalogue])),len(json.dumps(experiences))/10)
        experiences[-1]['assumptions'][0]['statement']='incompatible assumption with reused ID'
        with self.assertRaisesRegex(ContractError,'collision'):experience_prompt(experiences)

    def test_round_retrieval_keeps_grounding_for_every_graph_paper(self):
        records=[]
        for i in range(35):
            p=copy.deepcopy(papers()[0]);p['paper_id']=f'fixture:coverage:{i}'
            p['sections']={'abstract':f'Historical baseline {i}. '+('Agent authorization mechanism. '*42),
                           'full_text':f'Method {i}. '+('Evidence supporting causal comparison. '*32)}
            records.append(paper_record(p))
        snapshot=Snapshot({'cutoff':self.config['cutoff'],'papers':records,'excluded_future_versions':[]})
        selected={'selected_keys':list(snapshot.papers), 'extractions':{
            k:{'evidence_ids':list(snapshot.spans(k))} for k in snapshot.papers},'limitations':[]}
        graph={'graph_id':'fixture-graph','stages':[],'edges':[],'coverage_gaps':[]}
        base_context=graph_context(graph,snapshot,selected)
        self.assertEqual({e['paper_key'] for e in base_context['evidence'].values()},set(snapshot.papers))
        current=round_context(base_context,snapshot,self.config,None,[])
        self.assertLess(len(current['round_retrieval']['paper_keys']),len(records))
        self.assertTrue(set(base_context['evidence'])<=set(current['evidence']))
        self.assertEqual({e['paper_key'] for e in current['evidence'].values()},set(snapshot.papers))

    def test_later_round_cannot_ignore_experience(self):
        class Ignoring(SyntheticEndpoint):
            def respond(self,role,data):
                r=super().respond(role,data)
                if role=='propose' and data['round']>1:r['learning_trace']=[]
                return r
        with self.assertRaisesRegex(ContractError,'learning trace'):
            self.execute(Ignoring(high_value=False))
        self.assertEqual(len(unseal(self.directory()/'state.json')['rounds']),2)

    def test_dismissing_evidenced_conflict_requires_evidence(self):
        class Dismissal(SyntheticEndpoint):
            def respond(self,role,data):
                r=super().respond(role,data)
                if role=='critique':
                    r['objections'][0].update(kind='evidenced_conflict',evidence_ids=list(data['historical_context']['evidence'])[:1])
                if role=='assess':r['resolutions'][0].update(disposition='not_applicable',conditions_match=False,evidence_ids=[])
                return r
        with self.assertRaisesRegex(ContractError,'dismissal/resolution'):
            self.execute(Dismissal())

    def test_temporal_candidate_pool_covers_history_and_prioritizes_recent_two_years(self):
        records=[]
        for year in range(2016,2027):
            for i in range(25):
                p=copy.deepcopy(papers()[0]);p.update(paper_id=f'fixture:{year}-{i}',first_public_at=f'{year}-01-01T00:00:00Z',version_public_at=f'{year}-01-01T00:00:00Z')
                records.append(paper_record(p))
        snapshot=Snapshot({'cutoff':'2026-10-12T00:00:00Z','papers':records,'excluded_future_versions':[]})
        # Worst case observed in live retrieval: newest-year papers dominate all
        # unconstrained scores; the preceding recent year still needs exposure.
        chosen=temporal_rank(sorted(snapshot.papers,reverse=True),snapshot,30,self.config['retrieval'])
        years=[int(snapshot.get(k)['first_public_at'][:4]) for k in chosen]
        self.assertTrue(set(range(2016,2025))<=set(years))
        self.assertGreaterEqual(sum(y>=2025 for y in years),20)
        self.assertGreaterEqual(years.count(2025),10)
        self.assertGreaterEqual(years.count(2026),10)

    def test_fulltext_evidence_view_is_bounded_and_exact(self):
        p=copy.deepcopy(papers()[0]);p['sections']['full_text']=('Method: permission mechanism. '+('context '*100)+'\n\n')*100
        snapshot=Snapshot({'cutoff':self.config['cutoff'],'papers':[paper_record(p)],'excluded_future_versions':[]})
        view=evidence_view(snapshot,'fixture:1@v1','permission mechanism',4000)
        self.assertLessEqual(sum(len(s['quote']) for s in view.values()),4000)
        for span in view.values():
            self.assertEqual(span['quote'],p['sections'][span['section']][span['start']:span['end']])

    def test_pdf_source_bytes_are_archived_and_tampering_is_detected(self):
        archive=Archive(self.root/'archive');raw=b'%PDF-SYNTHETIC-NOT-A-PAPER'
        with patch('paperfuture.dataset.urllib.request.urlopen',return_value=io.BytesIO(raw)):
            result,path=archive.fetch('https://example.invalid/paper.pdf',pdf=True)
        self.assertEqual(result,raw)
        self.assertEqual(read_json(path.with_suffix('.pdf.json'))['sha256'],hashlib.sha256(raw).hexdigest())
        replay=Archive(self.root/'archive',replay=True)
        with patch('paperfuture.dataset.urllib.request.urlopen',side_effect=AssertionError('network')):
            self.assertEqual(replay.fetch('https://example.invalid/paper.pdf',pdf=True)[0],raw)
        path.write_bytes(raw+b'tampered')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            replay.fetch('https://example.invalid/paper.pdf',pdf=True)
