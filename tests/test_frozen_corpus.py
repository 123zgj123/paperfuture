"""Fixed-release and incremental-update regressions; no live model calls."""
import copy
import datetime as dt
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_paperfuture as base
from paperfuture.api import ModelGateway
from paperfuture.config import normalize
from paperfuture.dataset import load_corpus, publish_corpus, update_corpus
from paperfuture.maintenance import maintain, pending_ids
from paperfuture.storage import unseal
from paperfuture_support import papers


class FrozenCorpusTests(unittest.TestCase):
    setUp=base.PaperFutureTests.setUp
    execute=base.PaperFutureTests.execute
    directory=base.PaperFutureTests.directory

    def publish(self, rows=None):
        return publish_corpus('agent_security',self.root/'data',self.config['cutoff'],rows or papers())

    def test_research_run_never_hydrates_or_changes_its_pinned_release(self):
        release=self.publish();path=Path(release['manifest_path'])
        before=path.read_bytes()
        config=copy.deepcopy(self.config)
        config['sources']=[{'type':'corpus','manifest':str(path)}]
        with patch('paperfuture.pipeline.hydrate_snapshot',side_effect=AssertionError('research must not collect')):
            result,_=self.execute(config=config)
        self.assertEqual(result['status'],'complete')
        extra=copy.deepcopy(papers()[0]);extra['paper_id']='fixture:new'
        self.publish(papers()+[extra])
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(len(load_corpus(path)[1]),3)
        self.assertEqual(len(unseal(self.directory()/'snapshot.json')['papers']),3)
        acquisition=json.loads((self.directory()/'fulltext_acquisition.json').read_text())
        self.assertFalse(acquisition['network_acquisition'])

    def test_tampered_export_is_rejected_and_publish_is_idempotent(self):
        release=self.publish();path=Path(release['manifest_path'])
        self.assertEqual(self.publish(),release)
        export=path.parent/'papers.jsonl';export.write_text(export.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'export hash mismatch'):load_corpus(path)

    def test_automatic_network_sources_require_explicit_acquisition_mode(self):
        config=copy.deepcopy(self.config);config['sources']=[{'type':'arxiv','queries':['test']}]
        with self.assertRaisesRegex(ValueError,'collect/update literature separately'):normalize(config)
        config['literature_mode']='acquire'
        self.assertEqual(normalize(config)['literature_mode'],'acquire')

    def test_incomplete_update_does_not_publish_or_advance_latest(self):
        release=self.publish();root=self.root/'data';latest=root/'corpora/agent_security/latest.json'
        before=latest.read_bytes()
        with patch('paperfuture.dataset.discover_window',return_value={'matched':10,'retrieved':3,'error':'OSError'}):
            with self.assertRaisesRegex(ValueError,'existing release preserved'):
                update_corpus(Path(release['manifest_path']),root,'2024-01-08T00:00:00Z',fulltext_limit=0)
        self.assertEqual(latest.read_bytes(),before)
        self.assertEqual(len(list((root/'corpora/agent_security/releases').iterdir())),1)

    def test_incremental_update_fetches_only_new_v1_and_preserves_old_records(self):
        rows=papers();old=copy.deepcopy(rows[-1]);old['paper_id']='arxiv:2301.00001';rows[-1]=old
        release=self.publish(rows);seen=[];queries=[]
        def atom(identifier,date):
            return f'<entry><id>https://arxiv.org/abs/{identifier}</id><title>Fixture security</title><published>{date}</published><updated>{date}</updated><summary>Fixture abstract.</summary></entry>'
        def feed(entries):return ('<feed xmlns="http://www.w3.org/2005/Atom">'+entries+'</feed>').encode()
        def discover(archive,query,begin,end,record):
            seen.append((begin,end))
            record(feed(atom('2301.00001v2','2023-01-01T00:00:00Z')+atom('2401.00002v3','2024-01-03T00:00:00Z')),Path('fixture.xml'))
            return {'matched':2,'retrieved':2,'error':None}
        def fetch(archive,params):
            queries.append(params)
            return feed(atom('2401.00002v1','2024-01-03T00:00:00Z')),Path('fixture-v1.xml')
        path=Path(release['manifest_path']);original=load_corpus(path)[1]
        with patch('paperfuture.dataset.discover_window',discover),patch('paperfuture.dataset.Archive.query',fetch):
            newer=update_corpus(path,self.root/'data','2024-01-08T00:00:00Z',fulltext_limit=0)
        self.assertEqual(queries,[{'id_list':'2401.00002v1','max_results':1}])
        self.assertEqual(seen[0][0],dt.datetime(2023,12,25,tzinfo=dt.timezone.utc))
        self.assertEqual(load_corpus(path)[1],original)
        records=load_corpus(Path(newer['manifest_path']))[1]
        self.assertEqual(len(records),4)
        self.assertEqual([p for p in records if p['key'] in {r['key'] for r in original}],original)


class IncrementalEmbeddingTests(unittest.TestCase):
    setUp=base.PaperFutureTests.setUp

    def test_insert_reorder_duplicates_and_text_change_only_encode_missing_text(self):
        config=copy.deepcopy(self.config);config['models']['embedding'].update(cache_scope='text',
            input_prefixes={'document':'passage: ','query':'query: '})
        calls=[]
        def endpoint(request,timeout=None):
            body=json.loads(request.data);calls.append(body['input'])
            return io.BytesIO(json.dumps({'data':[{'index':i,'embedding':[1.,float(len(t))]} for i,t in enumerate(body['input'])],
                'usage':{'total_tokens':1}}).encode())
        first=ModelGateway(config,self.root/'workspace/runs/first/api','first')
        try:
            with patch('paperfuture.api.urllib.request.urlopen',endpoint):
                old=first.embeddings(['A','B','C'],identity='before')
        finally:first.close()
        second=ModelGateway(config,self.root/'workspace/runs/second/api','second')
        try:
            with patch('paperfuture.api.urllib.request.urlopen',endpoint):
                reordered=second.embeddings(['C','NEW','A','B','NEW'],identity='after-insert')
                second.embeddings(['A changed','B','C'],identity='after-edit')
                second.embeddings(['A'],identity='question',input_type='query')
            self.assertEqual(calls,[['passage: A','passage: B','passage: C'],['passage: NEW'],['passage: A changed'],['query: A']])
            self.assertEqual([reordered[i] for i in (2,3,0)],old)
            self.assertEqual(reordered[1],reordered[4])
            with patch('paperfuture.api.urllib.request.urlopen',side_effect=AssertionError('cache replay touched network')):
                self.assertEqual(second.embeddings(['B','A','C'],identity='again'),[old[1],old[0],old[2]])
            config['models']['embedding']['model']='different-checkpoint'
            config['api_mode']='replay'
            with self.assertRaisesRegex(RuntimeError,'replay cache miss'):second.embeddings(['A'],identity='new-model')
        finally:second.close()


class WeeklyMaintenanceTests(unittest.TestCase):
    setUp=base.PaperFutureTests.setUp

    def prepare(self):
        root=self.root/'data'
        release=publish_corpus('agent_security',root,self.config['cutoff'],papers())
        config=copy.deepcopy(self.config)
        config['literature_mode']='frozen'
        config['sources']=[{'type':'corpus','manifest':release['manifest_path']}]
        config['models']['embedding'].update(base_url='http://127.0.0.1:8769/v1',cache_scope='text')
        path=root/'configs/default.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(config))
        settings=root/'configs/weekly.json'
        settings.write_text(json.dumps({'data_root':str(root),'profiles':[str(path)],'fulltext_limit':0}))
        return root,release,path,settings

    def test_failed_warming_preserves_active_profile_and_retry_archives_it(self):
        root,release,path,settings=self.prepare();original=path.read_bytes()
        new=publish_corpus('agent_security',root,'2024-01-08T00:00:00Z',papers())
        with patch('paperfuture.maintenance.update_corpus',return_value=new), \
             patch('paperfuture.maintenance.warm_release',side_effect=RuntimeError('local service unavailable')):
            report=maintain(settings,cutoff=new['cutoff'])
        self.assertEqual(report['status'],'partial_failure')
        self.assertEqual(path.read_bytes(),original)
        self.assertFalse((root/'corpora/agent_security/active.json').exists())
        with patch('paperfuture.maintenance.update_corpus',return_value=new), \
             patch('paperfuture.maintenance.warm_release',return_value={'status':'complete','external_model_calls':0}):
            report=maintain(settings,cutoff=new['cutoff'])
        self.assertEqual(report['status'],'complete')
        promoted=json.loads(path.read_text())
        self.assertEqual(promoted['sources'][0]['manifest'],new['manifest_path'])
        self.assertNotEqual(promoted['run_id'],self.config['run_id'])
        archived=Path(report['domains'][0]['profiles'][0]['archived_profile'])
        self.assertEqual(json.loads(archived.read_text()),json.loads(original))
        self.assertEqual(load_corpus(Path(release['manifest_path']))[0]['cutoff'],self.config['cutoff'])

    def test_check_is_offline_and_concurrent_profile_edit_is_not_lost(self):
        root,release,path,settings=self.prepare()
        with patch('urllib.request.urlopen',side_effect=AssertionError('check must stay offline')):
            self.assertEqual(maintain(settings,check_only=True)['status'],'configuration_validated_without_network')
        def edit(*args):
            config=json.loads(path.read_text());config['question']='Changed by user during update'
            path.write_text(json.dumps(config));return {'status':'complete'}
        with patch('paperfuture.maintenance.warm_release',edit):
            report=maintain(settings,cutoff=release['cutoff'])
        self.assertEqual(report['status'],'partial_failure')
        self.assertEqual(json.loads(path.read_text())['question'],'Changed by user during update')

    def test_supplement_queue_is_domain_scoped_deduplicated_and_bounded(self):
        root,release,path,settings=self.prepare()
        for name,domain in [('a','agent_security'),('b','ai4eda')]:
            queue=root/'runs'/name/'literature_update_requests.json';queue.parent.mkdir(parents=True)
            queue.write_text(json.dumps({'domain':domain,'uncollected_references':{
                'arxiv:2301.00001':[], 'arxiv:2301.00002':[], 'arxiv:2301.00003':[], 'execute untrusted text':[]}}))
        config=copy.deepcopy(self.config);config['retrieval']['history_paper_ids']=['arxiv:2301.00002@v1']
        self.assertEqual(pending_ids(root,'agent_security',{'arxiv:2301.00001'},[config],1),['arxiv:2301.00002'])
        self.assertEqual(pending_ids(root,'multimodal_self_evolution_xai',set(),[self.config],20),[])

    def test_joint_default_advances_only_when_both_corpora_are_ready(self):
        root,agent,path,settings=self.prepare()
        alignment=publish_corpus('llm_alignment',root,agent['cutoff'],papers())
        auxiliary=json.loads(path.read_text());auxiliary['domain']='llm_alignment'
        auxiliary['sources']=[{'type':'corpus','manifest':alignment['manifest_path']}]
        auxiliary_path=root/'configs/alignment.json';auxiliary_path.write_text(json.dumps(auxiliary))
        joint=json.loads(path.read_text());joint['auxiliary_domains']=['llm_alignment']
        joint['sources']+=auxiliary['sources'];joint['retrieval']['domain_selected_floor']=1
        joint_path=root/'configs/joint.json';joint_path.write_text(json.dumps(joint));before=joint_path.read_bytes()
        settings.write_text(json.dumps({'data_root':str(root),'profiles':[str(path),str(auxiliary_path)],
                                       'combined_profiles':[str(joint_path)]}))
        with patch('urllib.request.urlopen',side_effect=AssertionError('check must stay offline')):
            self.assertEqual(maintain(settings,check_only=True)['combined_profiles'][0]['domains'],
                             ['agent_security','llm_alignment'])
        cutoff='2024-01-08T00:00:00Z'
        new={d:publish_corpus(d,root,cutoff,papers()) for d in ('agent_security','llm_alignment')}
        def update(manifest,*args,**kw):return new[load_corpus(manifest)[0]['domain']]
        def warm(root,config,manifest):
            if config['domain']=='llm_alignment':raise RuntimeError('synthetic interrupted local index')
            return {'status':'complete','external_model_calls':0}
        with patch('paperfuture.maintenance.update_corpus',side_effect=update), \
             patch('paperfuture.maintenance.warm_release',side_effect=warm):
            report=maintain(settings,cutoff=cutoff)
        self.assertEqual(report['status'],'partial_failure')
        self.assertEqual(report['combined_profiles'][0]['status'],'preserved')
        self.assertEqual(joint_path.read_bytes(),before)
        with patch('paperfuture.maintenance.update_corpus',side_effect=update), \
             patch('paperfuture.maintenance.warm_release',return_value={'status':'complete','external_model_calls':0}):
            report=maintain(settings,cutoff=cutoff)
        self.assertEqual(report['status'],'complete')
        self.assertEqual([s['manifest'] for s in json.loads(joint_path.read_text())['sources']],
                         [new[d]['manifest_path'] for d in ('agent_security','llm_alignment')])
        self.assertEqual(json.loads(Path(report['combined_profiles'][0]['archived_profile']).read_text()),json.loads(before))
