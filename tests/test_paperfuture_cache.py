"""Receipt-backed cache maintenance, independent of scientific model judgments."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from paperfuture.storage import digest, seal, unseal

spec=importlib.util.spec_from_file_location('repack_embeddings',Path(__file__).resolve().parents[1]/'tools/repack_embedding_cache.py')
cache=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class EmbeddingCacheRepackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.target=self.root/'runs/target'
        self.model={'base_url':'https://fixture.invalid/v1','model':'fixture-embedding'}
        seal(self.target/'spec.json',{'models':{'embedding':self.model}})
        self.papers=[{'key':f'paper:{i:03}','title':f'title {i}','sections':{'abstract':f'abstract {i}'}} for i in range(40)]
        self.texts=[p['title']+'\n'+p['sections']['abstract'] for p in self.papers]
        self.vectors=[[1.0,float(i+1)] for i in range(40)]
        seal(self.target/'snapshot.json',{'papers':self.papers})

    def receipt(self,name,texts,vectors,model=None):
        directory=self.root/'runs'/name/'api';directory.mkdir(parents=True)
        body={'model':model or self.model['model'],'input':texts,'encoding_format':'float'}
        request={'role':'embedding','url':self.model['base_url']+'/embeddings','body':body}
        result={'data':[{'index':i,'embedding':v} for i,v in reversed(list(enumerate(vectors)))]}
        receipt={'request':request,'raw':json.dumps(result)}
        rid=digest(request);seal(directory/'receipts'/(rid+'.json'),receipt)
        c=sqlite3.connect(directory/'usage.sqlite')
        c.execute('create table attempts(request_id text,role text,status text)')
        c.execute('insert into attempts values(?,?,?)',(rid,'embedding','received'));c.commit();c.close()
        return receipt

    def test_repacking_changed_batches_copies_exact_rows_with_provenance(self):
        self.receipt('old-odd',self.texts[1::2],self.vectors[1::2])
        self.receipt('old-even',self.texts[::2],self.vectors[::2])
        result=cache.repack(self.root,self.target)
        self.assertEqual(len(result['created_batches']),2)
        self.assertEqual(result['additional_api_calls'],0)
        for row in result['created_batches']:
            saved=unseal(Path(row['cache_path']));offset=row['offset']
            self.assertEqual(saved['vectors'],self.vectors[offset:offset+32])
            self.assertEqual(saved['input_hash'],digest(self.texts[offset:offset+32]))
            self.assertEqual(len(saved['provenance']['rows']),len(saved['vectors']))
        self.assertEqual(cache.repack(self.root,self.target)['created_batches'],[])

    def test_conflicting_vectors_and_other_models_are_never_silently_reused(self):
        self.receipt('valid',self.texts,self.vectors)
        self.receipt('conflict',[self.texts[0]],[[2.0,3.0]])
        self.receipt('foreign',[self.texts[32]],[[8.0,9.0]],model='another-model')
        result=cache.repack(self.root,self.target)
        self.assertEqual(result['conflicting_texts_excluded'],1)
        self.assertEqual([r['offset'] for r in result['created_batches']],[32])
        self.assertEqual(result['batches_still_requiring_provider'],[0])

    def test_invalid_indices_are_rejected_before_vector_reuse(self):
        receipt=self.receipt('bad',self.texts[:2],self.vectors[:2])
        data=json.loads(receipt['raw']);data['data'][1]['index']=1
        receipt['raw']=json.dumps(data)
        with self.assertRaisesRegex(ValueError,'indexed'):
            cache.receipt_rows(receipt,self.model['base_url']+'/embeddings',self.model['model'])
