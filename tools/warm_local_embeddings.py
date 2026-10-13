"""Precompute title/abstract vectors using only a loopback embedding service.

This is cache maintenance, not an idea-generation experiment. It never invokes
a chat model. Successful receipts and exact per-row reuse remain on the data
volume; rerunning resumes the same immutable input snapshot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paperfuture.api import ModelGateway
from paperfuture.config import load_config
from paperfuture.storage import atomic_json, digest, exclusive, seal, unseal
from repack_embedding_cache import repack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    model=config['models']['embedding']
    if urlparse(model['base_url']).hostname != '127.0.0.1':
        raise ValueError('warm_local_embeddings only allows the local 127.0.0.1 embedding service')
    papers={}
    for source in config['sources']:
        if source['type']!='jsonl':
            raise ValueError('precompute requires an already exported JSONL dataset')
        path=Path(source['path']).resolve()
        if not path.is_relative_to(args.data_root.resolve()):
            raise ValueError('paper dataset must be on the declared data volume')
        for line in path.read_text().splitlines():
            if not line.strip():continue
            p=json.loads(line)
            if 'abstract' not in p['sections']:
                raise ValueError('precompute requires explicit abstract text')
            value={'key':p['key'],'title':p['title'],'sections':{'abstract':p['sections']['abstract']}}
            if p['key'] in papers and papers[p['key']]!=value:
                raise ValueError('conflicting paper text under the same version key')
            papers[p['key']]=value
    ordered=[papers[k] for k in sorted(papers)]
    if not ordered:raise ValueError('empty dataset')
    snapshot={'papers':ordered}
    identity=digest({'model':model,'input':snapshot})
    directory=args.data_root/'runs'/('local-embedding-index-'+config['domain']+'-'+identity[:12])
    config['api'].update(max_calls=1000,max_tokens=100_000_000,timeout_seconds=600)
    with exclusive(directory/'.lock'):
        if (directory/'snapshot.json').exists():
            if unseal(directory/'snapshot.json')!=snapshot:raise ValueError('immutable input mismatch')
        else:
            seal(directory/'snapshot.json',snapshot)
            seal(directory/'spec.json',{'artifact_type':'local_embedding_cache_maintenance_not_research_run',
                'models':{'embedding':model},'replicate_id':'local-embedding-index','input_hash':digest(snapshot)})
        reuse=repack(args.data_root,directory)
        atomic_json(directory/'repacking.json',reuse)
        gateway=ModelGateway(config,directory/'api',identity)
        started=time.monotonic()
        try:
            for offset in range(0,len(ordered),32):
                batch=ordered[offset:offset+32]
                gateway.embeddings([p['title']+'\n'+p['sections']['abstract'] for p in batch],
                    identity=f'documents:{identity}:{offset}')
                progress={'complete_papers':min(offset+32,len(ordered)),'total_papers':len(ordered),
                          'elapsed_seconds':round(time.monotonic()-started,3)}
                atomic_json(directory/'progress.json',progress)
                print(json.dumps(progress),flush=True)
            gateway.embeddings([config['question']],identity=f'query:{identity}',input_type='query')
            report={**progress,'status':'complete','external_api_calls':0,'model':model['model'],
                    'input_hash':digest(snapshot),'usage':gateway.usage(),'embedding_scope':'title_and_abstract'}
            atomic_json(directory/'completed.json',report)
            print(json.dumps({'status':'complete','directory':str(directory),'papers':len(ordered)}),flush=True)
        finally:
            gateway.close()


if __name__=='__main__':
    main()
