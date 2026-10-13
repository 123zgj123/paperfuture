"""Local embedding protocol, cache separation, and nontruncation checks."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_paperfuture as base
from paperfuture.api import ModelGateway
from paperfuture.config import normalize

spec=importlib.util.spec_from_file_location('local_embedding_server',Path(__file__).resolve().parents[1]/'tools/local_embedding_server.py')
server=importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class LocalEmbeddingTests(unittest.TestCase):
    setUp=base.PaperFutureTests.setUp

    def test_query_document_prefixes_have_distinct_verbatim_cache_entries(self):
        config=copy.deepcopy(self.config)
        config['models']['embedding']['input_prefixes']={'query':'query: ','document':'passage: '}
        config=normalize(config);calls=[]
        def endpoint(request,timeout=None):
            data=json.loads(request.data);calls.append(data)
            return io.BytesIO(json.dumps({'data':[{'index':i,'embedding':[1.,float(len(t))]} for i,t in enumerate(data['input'])],
                                          'usage':{'total_tokens':5}}).encode())
        directory=self.root/'workspace/runs/local/api'
        gateway=ModelGateway(config,directory,'local')
        try:
            with patch('paperfuture.api.urllib.request.urlopen',endpoint):
                document=gateway.embeddings(['same text'],identity='documents:1')
                query=gateway.embeddings(['same text'],identity='query:1',input_type='query')
                self.assertEqual(gateway.embeddings(['same text'],identity='documents:2'),document)
            self.assertEqual([c['input'] for c in calls],[['passage: same text'],['query: same text']])
            self.assertNotEqual(query,document)
            config['api_mode']='replay'
            with patch('paperfuture.api.urllib.request.urlopen',side_effect=AssertionError('network')):
                self.assertEqual(gateway.embeddings(['same text'],identity='query:2',input_type='query'),query)
            config['models']['embedding']['input_prefixes']['document']='changed: '
            with self.assertRaisesRegex(RuntimeError,'replay cache miss'):
                gateway.embeddings(['same text'],identity='documents:3')
        finally:
            gateway.close()

    def test_prefix_options_are_embedding_only_and_explicit(self):
        for role,value in [('default',{'query':'q','document':'d'}),('embedding',{'query':'q'}),
                           ('embedding',{'query':None,'document':'d'})]:
            config=copy.deepcopy(self.config);config['models'][role]['input_prefixes']=value
            with self.assertRaisesRegex(ValueError,'input_prefixes'):normalize(config)

    def test_local_endpoint_rejects_untyped_or_wrong_model_inputs(self):
        for payload in [{'model':'wrong','input':['query: x']}, {'model':'local','input':['x']},
                        {'model':'local','input':['query: ']}, {'model':'local','input':['query: x'],'encoding_format':'base64'}]:
            with self.assertRaises(ValueError):server.validate_inputs(payload,'local')
        self.assertEqual(server.validate_inputs({'model':'local','input':['query: x','passage: y']},'local'),['query: x','passage: y'])

    def test_overlength_input_is_rejected_before_model_inference(self):
        class Fake:
            def tokenizer(self,*a,**kw):
                assert kw['truncation'] is False
                return {'input_ids':[[1,2,3]]}
            def encode(self,*a,**kw):raise AssertionError('overlength input reached model')
        encoder=server.Encoder.__new__(server.Encoder);encoder.model=Fake();encoder.max_tokens=2
        with self.assertRaisesRegex(ValueError,'chunk it explicitly'):encoder.encode(['query: long'])

    def test_modified_checkpoint_is_rejected_before_loading(self):
        path=self.root/'config.json';path.write_text('{}')
        manifest={'files':[{'path':'config.json','bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest(),'upstream_hash_verified':True}]}
        (self.root/'download-manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(server.verify_checkpoint(self.root),manifest)
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError,'hash verification'):server.verify_checkpoint(self.root)
