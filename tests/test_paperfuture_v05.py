"""Cross-corpus retrieval, explicit idea isolation and bounded research tools."""
import copy
import io
import json
from pathlib import Path
import sys
import unittest
import urllib.error
from unittest.mock import patch

import test_paperfuture as base
from paperfuture.config import normalize
from paperfuture.contracts import ContractError
from paperfuture.dataset import Archive, discover_facets, load_corpus, publish_corpus, publish_source, published_record_reuse, supplement_corpus
from paperfuture.exploration import experiences_from_rounds
from paperfuture.literature import LiteratureStore, Snapshot, corpus_scope, round_context
from paperfuture.memory import round_scope, validate_transfers
from paperfuture.research_skills import apply_tool_plan, validate_tool_plan
from paperfuture.storage import unseal
from paperfuture_support import papers, SyntheticEndpoint


class V05Tests(unittest.TestCase):
    setUp=base.PaperFutureTests.setUp
    execute=base.PaperFutureTests.execute
    directory=base.PaperFutureTests.directory

    def combined(self):
        rows=papers();config=copy.deepcopy(self.config)
        primary=publish_corpus('agent_security',self.root/'data',config['cutoff'],rows[:2])
        auxiliary=publish_corpus('llm_alignment',self.root/'data',config['cutoff'],rows[1:])
        config['sources']=[{'type':'corpus','manifest':r['manifest_path']} for r in (primary,auxiliary)]
        config['auxiliary_domains']=['llm_alignment']
        config['retrieval'].update(domain_candidate_floor=2,domain_selected_floor=1)
        return normalize(config)

    def test_joint_retrieval_preserves_both_sources_and_deduplicates_overlap(self):
        result,endpoint=self.execute(config=self.combined())
        snapshot=unseal(self.directory()/'snapshot.json')
        self.assertEqual(len(snapshot['papers']),3)
        self.assertEqual(snapshot['paper_domains']['fixture:2@v1'],['agent_security','llm_alignment'])
        selected=unseal(self.directory()/'selected.json')
        self.assertGreaterEqual(selected['domain_selected_counts']['agent_security'],1)
        self.assertGreaterEqual(selected['domain_selected_counts']['llm_alignment'],1)
        context=unseal(self.directory()/'rounds/01/evidence_context.json')
        self.assertTrue(all(n>=1 for n in context['round_retrieval']['domain_counts'].values()))
        document_calls=[data['input'] for role,data in endpoint.requests if role=='embedding' and len(data['input'])>1]
        self.assertEqual(len(document_calls),1)
        self.assertEqual(len(document_calls[0]),3)
        self.assertEqual(result['completed_rounds'],1)

    def test_undeclared_or_missing_auxiliary_library_is_rejected(self):
        config=self.combined();config['auxiliary_domains']=[]
        with self.assertRaisesRegex(ValueError,'undeclared research domain'):self.execute(config=config)
        config=self.combined();config['run_id']='missing-aux';config['sources']=config['sources'][:1]
        with self.assertRaisesRegex(ValueError,'supply every declared corpus'):self.execute(config=config)

    def test_round_domain_allocation_survives_late_query_matches_and_earlier_cutoff(self):
        rows=papers()
        rows[2]['title']='SYNTHETIC rewardmarker'
        rows[2]['sections']={'abstract':'SYNTHETIC rewardmarker distinct alignment objective.'}
        primary=publish_corpus('agent_security',self.root/'data',self.config['cutoff'],rows[:2])
        # Future metadata belongs in the archived release, never model context.
        future=copy.deepcopy(rows[2]);future['paper_id']='fixture:future'
        future.update(first_public_at='2025-01-01T00:00:00Z',version_public_at='2025-01-01T00:00:00Z')
        auxiliary=publish_corpus('llm_alignment',self.root/'data','2025-02-01T00:00:00Z',[rows[2],future])
        config=copy.deepcopy(self.config);config.update(question='rewardmarker',auxiliary_domains=['llm_alignment'],
            sources=[{'type':'corpus','manifest':r['manifest_path']} for r in (primary,auxiliary)])
        config['retrieval'].update(round_papers=2,domain_selected_floor=1)
        config=normalize(config)
        store=LiteratureStore(self.root/'scope.sqlite');self.addCleanup(store.close)
        snapshot=corpus_scope(store.freeze(store.add_many(rows+[future]),config['cutoff']),config)
        base_context={'complete_stage_graph':{'stages':[],'edges':[]},'evidence':{},'papers':[]}
        feedback={'action':'switch','navigation':{'queries':{'alternative':'security',
            'bottleneck':'stage 1','falsifier':'stage 2'},'target_stage_ids':[],'experience_ids':[]}}
        context=round_context(base_context,snapshot,config,feedback,[])
        self.assertEqual(context['round_retrieval']['domain_counts'],{'agent_security':1,'llm_alignment':1})
        self.assertNotIn('fixture:future@v1',snapshot.papers)
        from paperfuture.graph import graph_context
        selected={'selected_keys':list(snapshot.papers),'extractions':{k:{'evidence_ids':[]} for k in snapshot.papers},'limitations':[]}
        model_context=graph_context({'stages':[],'edges':[],'coverage_gaps':[],'graph_id':'synthetic'},snapshot,selected)
        alignment=next(r for r in model_context['corpus_releases'] if r['domain']=='llm_alignment')
        self.assertEqual(alignment['eligible_papers'],1)
        self.assertNotIn('papers',alignment)
        self.assertNotIn('cutoff',alignment)

    def test_new_experiment_has_no_previous_idea_or_simulation_memory(self):
        self.execute()
        config=copy.deepcopy(self.config);config['run_id']='independent'
        result,endpoint=self.execute(config=config)
        request=next(data for role,data in endpoint.requests if role=='propose')
        self.assertEqual(request['cumulative_experiences'],[])
        self.assertEqual(request['previous_candidates'],[])
        self.assertEqual(request['memory_scope']['run_id'],'independent')
        old=unseal(self.directory()/'rounds/01/evaluated.json')
        new=unseal(self.root/'workspace/runs/independent/rounds/01/evaluated.json')
        with self.assertRaisesRegex(ContractError,'foreign or unowned'):
            round_scope('independent',2,{'action':'revise','parent_round':1},[old])
        with self.assertRaisesRegex(ContractError,'mixed run ownership'):
            experiences_from_rounds([old,{**new,'number':2}])

    def test_switch_only_expands_requested_lessons_and_requires_transfer(self):
        class Switch(SyntheticEndpoint):
            def respond(self,role,data):
                value=super().respond(role,data)
                if role=='assess':
                    value['suggested_action']='switch'
                    value['navigation']['experience_ids']=[data['cumulative_experiences'][-1]['id']]
                return value
        config=copy.deepcopy(self.config);config['max_rounds']=3;config['research_skills']['enabled']=True
        result,endpoint=self.execute(Switch(high_value=False),config)
        scopes=[unseal(self.directory()/f'rounds/{i:02}/memory_scope.json') for i in (1,2,3)]
        self.assertEqual(len({s['branch_id'] for s in scopes}),3)
        self.assertTrue(all(s['ancestor_idea_ids']==[] for s in scopes))
        for i in (2,3):
            rd=self.directory()/f'rounds/{i:02}'
            self.assertEqual(len(unseal(rd/'experience_memory.json')),1)
            self.assertGreater(len(unseal(rd/'experience_index.json')),0)
            self.assertEqual(len(unseal(rd/'proposal.json')['memory_transfers']),1)
        for role,data in endpoint.requests:
            if role=='propose' and data['round']>1:
                self.assertNotIn('parent_idea',data['conditional_feedback'])
                self.assertTrue(all('idea' not in item for item in data['previous_candidates']))
            if role=='research_tools' and data['conditional_navigation']:
                self.assertNotIn('parent_idea',data['conditional_navigation'])
                self.assertNotIn('parent_unknowns',data['conditional_navigation'])
        # Reconstruct the stored tool/memory inputs without any model calls.
        sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
        from audit_paperfuture import audit_run
        with patch('urllib.request.urlopen',side_effect=AssertionError('audit touched network')):
            audit_run(self.directory())

    def test_missing_cross_branch_applicability_record_stops_before_simulation(self):
        class Missing(SyntheticEndpoint):
            def respond(self,role,data):
                value=super().respond(role,data)
                if role=='assess':value['suggested_action']='switch'
                if role=='propose' and data['round']==2:value['memory_transfers']=[]
                return value
        endpoint=Missing(high_value=False)
        with self.assertRaisesRegex(ContractError,'explicit memory transfer'):self.execute(endpoint)
        self.assertEqual(sum(role=='simulate' for role,_ in endpoint.requests),1)
        self.assertFalse((self.directory()/'rounds/02/simulation.json').exists())

    def test_retained_condition_cannot_bypass_transfer_by_omitting_learning_trace(self):
        scope={'ancestor_idea_ids':['run/idea/01']}
        experiences=[{'id':'parent','source_idea_id':'run/idea/01','assumptions':[]},
            {'id':'foreign','source_idea_id':'run/idea/02','assumptions':[{'id':'foreign-premise'}]}]
        proposal={'learning_trace':[{'experience_id':'parent','usage':'retain'}],
            'assumptions':[], 'assumption_decisions':[{'assumption_id':'foreign-premise','disposition':'retain'}],
            'memory_transfers':[]}
        with self.assertRaisesRegex(ContractError,'retained cross-branch conditions'):
            validate_transfers(proposal,experiences,scope,{})
        proposal['learning_trace'].append({'experience_id':'foreign','usage':'not_applicable'})
        with self.assertRaisesRegex(ContractError,'retained cross-branch conditions'):
            validate_transfers(proposal,experiences,scope,{})
        proposal['learning_trace'][-1]['usage']='test'
        proposal['memory_transfers']=[{'experience_id':'foreign','applicability':'matched threat model',
            'condition_differences':['new intervention'],'test':'measure applicability under changed conditions','evidence_ids':[]}]
        validate_transfers(proposal,experiences,scope,{})

    def test_skill_search_read_and_compare_are_bounded_replayable_and_add_no_candidates(self):
        def request(tool,**kw):
            return {'tool':tool,'query':'agent security mechanism','domains':[], 'paper_keys':[],
                    'stage_ids':[],'purpose':'closest_prior_art','reason':'synthetic inspect source',**kw}
        class Inspect(SyntheticEndpoint):
            def respond(self,role,data):
                if role=='research_tools':
                    requests=([request('search_papers'),request('compare_stages',stage_ids=['s1','s2'])]
                        if not data['previous_tool_results'] else [request('read_passages',paper_keys=['fixture:1@v1'])])
                    return {'requests':requests,'stop_reason':'synthetic bounded source audit'}
                return super().respond(role,data)
        config=copy.deepcopy(self.config);config['research_skills']['enabled']=True
        result,endpoint=self.execute(Inspect(),config)
        roles=[r for r,_ in endpoint.requests]
        self.assertEqual(roles.count('research_tools'),2)
        self.assertEqual(roles.count('propose'),1)
        rd=self.directory()/'rounds/01'
        context=unseal(rd/'evidence_context.json')
        self.assertEqual(context['skill_tool_audit']['tool_calls'],3)
        self.assertLessEqual(context['skill_tool_audit']['additional_evidence_chars'],10000)
        self.assertEqual(unseal(rd/'skill_tools/01/results.json')[0]['method'],'BM25_per_domain_then_RRF')
        sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
        from audit_paperfuture import audit_run
        with patch('urllib.request.urlopen',side_effect=AssertionError('audit touched network')):
            audit_run(self.directory())
            replay,_=self.execute(config=config)
        self.assertEqual(result,replay)

    def test_tool_cannot_escape_snapshot_domain_or_budget(self):
        self.execute()
        snapshot=Snapshot(unseal(self.directory()/'snapshot.json'))
        context=unseal(self.directory()/'rounds/01/evidence_context.json')
        request={'tool':'read_passages','query':'method','domains':[],
                 'paper_keys':['other-run:secret@v1'],'stage_ids':[],'purpose':'mechanism','reason':'fixture'}
        plan={'requests':[request],'stop_reason':'fixture'}
        with self.assertRaisesRegex(ContractError,'tool paper scope'):validate_tool_plan(plan,snapshot,context,self.config,4)
        request.update(tool='search_papers',paper_keys=[],domains=['llm_alignment'])
        with self.assertRaisesRegex(ContractError,'tool corpus domains'):validate_tool_plan(plan,snapshot,context,self.config,4)
        request['domains']=[]
        with self.assertRaisesRegex(ContractError,'allowance exceeded'):validate_tool_plan(plan,snapshot,context,self.config,0)
        request['tool']='execute_shell'
        with self.assertRaises(ContractError):validate_tool_plan(plan,snapshot,context,self.config,4)

    def test_facet_union_deduplicates_provider_counts_and_keeps_incomplete_facet_visible(self):
        plan={'facets':{'a':'alpha','b':'beta'}}
        def discover(archive,query,begin,end,callback):
            callback(query,Path('fixture'))
            return {'matched':2,'retrieved':2,'error':None}
        record=lambda raw,path:[{'paper_id':k} for k in (['one','shared'] if raw=='alpha' else ['shared','two'])]
        import datetime as dt
        date=dt.datetime(2026,1,1,tzinfo=dt.timezone.utc)
        with patch('paperfuture.dataset.discover_window',discover):
            audit=discover_facets(None,plan,date,date,record)
        self.assertEqual(audit['matched'],3)
        self.assertEqual(sum(w['matched'] for w in audit['facets']),4)
        with patch('paperfuture.dataset.discover_window',return_value={'matched':3,'retrieved':1,'error':'TimeoutError'}):
            audit=discover_facets(None,plan,date,date,record)
        self.assertIsNone(audit['matched'])
        self.assertEqual(audit['error'],'incomplete_facet_discovery')

    def test_existing_version_reuse_requires_discovery_membership_and_honors_cutoff(self):
        rows=papers()[:2]
        rows[0]['paper_id']='arxiv:2101.00001';rows[1]['paper_id']='arxiv:2201.00001'
        release=publish_corpus('agent_security',self.root/'data',self.config['cutoff'],rows)
        wanted={'2101.00001v1','2201.00001v1','2301.99999v1'}
        with patch('urllib.request.urlopen',side_effect=AssertionError('reuse must not fetch')):
            reused,origins=published_record_reuse(self.root/'data',wanted,'2021-12-31T00:00:00Z')
        self.assertEqual(set(reused),{'2101.00001v1'})
        self.assertEqual(origins['2101.00001v1']['release_id'],release['release_id'])
        self.assertEqual(published_record_reuse(self.root/'data',set(),self.config['cutoff']),( {}, {} ))

    def test_arxiv_retry_after_is_not_shortened_and_persists_for_another_client(self):
        clock=[1000.];sent=[]
        def response(request,timeout):
            sent.append(clock[0])
            if len(sent)==1:
                raise urllib.error.HTTPError(request.full_url,429,'fixture limit',{'Retry-After':'120'},None)
            return io.BytesIO(b'<feed/>')
        archive=Archive(self.root/'archive')
        with patch('paperfuture.dataset.time.time',side_effect=lambda:clock[0]), \
             patch('paperfuture.dataset.time.sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)), \
             patch('urllib.request.urlopen',response):
            archive.fetch('https://example.invalid/fixture-one')
            (archive.root/'.cooldown.json').write_text(json.dumps({'not_before':clock[0]+180,'consecutive_429':2}))
            Archive(archive.root).fetch('https://example.invalid/fixture-two')
        self.assertGreaterEqual(sent[1]-sent[0],120)
        self.assertGreaterEqual(sent[2]-sent[1],180)

    def test_historical_supplement_preserves_cutoff_old_release_and_version_filter(self):
        rows=papers();root=self.root/'data'
        for i,p in enumerate(rows,1):p['paper_id']=f'arxiv:2{i}01.00001'
        base_release=publish_corpus('llm_alignment',root,self.config['cutoff'],rows[:2])
        future=copy.deepcopy(rows[2]);future.update(paper_id='arxiv:2601.00001',
            first_public_at='2026-01-01T00:00:00Z',version_public_at='2026-01-01T00:00:00Z')
        def enrich(p,archive):
            return {**p,'sections':{**p['sections'],'full_text':'Synthetic complete source passage. '*20}}
        with patch.object(Archive,'query',return_value=(b'fixture',Path('fixture'))), \
             patch('paperfuture.dataset.parse_arxiv',return_value=[rows[2],future]), \
             patch('paperfuture.dataset.enrich_fulltext',side_effect=enrich):
            release=supplement_corpus(Path(base_release['manifest_path']),root,
                paper_ids=['arxiv:2101.00001','arxiv:2301.00001','arxiv:2601.00001'],fulltext_limit=2)
        self.assertEqual(release['cutoff'],base_release['cutoff'])
        self.assertEqual(release['papers'],3)
        self.assertEqual(release['fulltext_papers'],2)
        self.assertNotEqual(release['release_id'],base_release['release_id'])
        self.assertEqual(len(load_corpus(Path(base_release['manifest_path']))[1]),2)
        self.assertEqual({p['paper_id'] for p in load_corpus(Path(release['manifest_path']))[1]},
                         {'arxiv:2101.00001','arxiv:2201.00001','arxiv:2301.00001'})

    def test_incomplete_gap_query_does_not_publish_a_partial_release(self):
        root=self.root/'data'
        release=publish_corpus('llm_alignment',root,self.config['cutoff'],papers())
        pointer=root/'corpora/llm_alignment/latest.json';before=pointer.read_bytes()
        with patch('paperfuture.dataset.discover_window',return_value={'matched':10,'retrieved':2,'error':'HTTPError429'}):
            with self.assertRaisesRegex(ValueError,'old corpus preserved'):
                supplement_corpus(Path(release['manifest_path']),root,queries=['missing mechanism'])
        self.assertEqual(pointer.read_bytes(),before)
        self.assertEqual(len(list((root/'corpora/llm_alignment/releases').glob('*/manifest.json'))),1)

    def test_unicode_line_separator_inside_json_string_is_not_a_paper_boundary(self):
        rows=papers()
        rows[0]['sections']['abstract']='SYNTHETIC source text with separator \u2028 and paragraph \u2029 and next line \u0085 preserved.'
        source=self.root/'unicode.jsonl'
        source.write_text('\n'.join(json.dumps(p,ensure_ascii=False) for p in rows)+'\n')
        release=publish_source('llm_alignment',self.root/'data',self.config['cutoff'],source)
        manifest,restored=load_corpus(Path(release['manifest_path']))
        self.assertEqual(manifest['papers'],3)
        self.assertEqual(restored[0]['sections']['abstract'],rows[0]['sections']['abstract'])
