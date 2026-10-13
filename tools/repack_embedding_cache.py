"""Repack verified independent embedding rows after corpus insertions.

No vectors are inferred or recomputed. Each output row is copied verbatim from
a successful provider receipt for the exact text, model and endpoint. This
maintenance command is opt-in because it assumes batch-independent embedding
semantics. It never overwrites a cache entry or a historical run artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paperfuture.api import ModelGateway, embedding_inputs
from paperfuture.storage import digest, seal, unseal, atomic_json, exclusive


def receipt_rows(receipt, url, model):
    request = receipt['request']
    if request.get('role') != 'embedding' or request.get('url') != url:
        return []
    body = request['body']
    if body.get('model') != model or set(body) != {'model', 'input', 'encoding_format'} or body['encoding_format'] != 'float':
        return []
    if not isinstance(body['input'], list) or not all(isinstance(t, str) for t in body['input']):
        return []
    result = json.loads(receipt['raw'])
    data = result.get('data', [])
    if sorted(r.get('index') for r in data) != list(range(len(body['input']))):
        raise ValueError('invalid indexed embedding receipt')
    vectors = [r['embedding'] for r in sorted(data, key=lambda r:r['index'])]
    ModelGateway._validate_vectors(vectors)
    return list(zip(body['input'], vectors))


def repack(root, directory):
    spec = unseal(directory/'spec.json')
    snapshot = unseal(directory/'snapshot.json')
    model = spec['models']['embedding']
    url = model['base_url']+'/embeddings'
    saved = {}; conflicts = set(); sources = 0
    # SQLite narrows reads to successful embedding receipts, excluding chats.
    for run in sorted((root/'runs').iterdir()):
        database = run/'api/usage.sqlite'
        if not database.is_file(): continue
        connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
        try:
            rows = connection.execute("SELECT request_id FROM attempts WHERE role='embedding' AND status='received'").fetchall()
        finally:
            connection.close()
        for (request_id,) in rows:
            path = run/'api/receipts'/(request_id+'.json')
            if not path.is_file(): continue
            receipt = unseal(path)
            if digest(receipt['request']) != request_id:
                raise ValueError('receipt request identity mismatch')
            pairs = receipt_rows(receipt, url, model['model'])
            if not pairs: continue
            sources += 1
            provenance = {'source_receipt':str(path), 'source_receipt_hash':digest(receipt)}
            for text, vector in pairs:
                # Any conflicting vector for an identical input abstains from
                # reuse; do not silently pick whichever receipt was read last.
                if text in saved and saved[text][0] != vector: conflicts.add(text)
                saved.setdefault(text, (vector, provenance))
    documents = {p['key']:p['title']+'\n'+p['sections'].get('abstract',next(iter(p['sections'].values()))) for p in snapshot['papers']}
    keys = sorted(documents); created=[]; skipped=[]
    for offset in range(0,len(keys),32):
        texts = embedding_inputs([documents[k] for k in keys[offset:offset+32]], model, "document")
        request = {'model':model['model'], 'input':texts, 'encoding_format':'float'}
        path = root/'cache/embeddings'/(digest({'base_url':model['base_url'],'request':request})+'.json')
        if path.exists(): continue
        if any(t not in saved or t in conflicts for t in texts):
            skipped.append(offset);continue
        vectors = [saved[t][0] for t in texts]
        # Provider dimension changes cannot be mixed across receipts.
        if len({len(v) for v in vectors}) != 1:
            skipped.append(offset);continue
        ModelGateway._validate_vectors(vectors)
        entries = [dict(saved[t][1],input_sha256=digest(t),vector_sha256=digest(saved[t][0])) for t in texts]
        if path.exists(): continue  # A concurrent pipeline may just have filled it.
        seal(path, {'input_hash':digest(texts), 'vectors':vectors, 'provenance':{
            'model':model['model'], 'base_url':model['base_url'],
            'method':'verbatim_row_repacking_from_verified_provider_receipts',
            'assumption':'embedding output for one text is independent of its batch companions',
            'rows':entries,'model_identity_independently_verified':False}})
        created.append({'offset':offset,'cache_path':str(path)})
    return {'target_run':directory.name,'model':model['model'],'verified_source_receipts':sources,
        'distinct_texts':len(saved),'conflicting_texts_excluded':len(conflicts),
        'created_batches':created,'batches_still_requiring_provider':skipped,'additional_api_calls':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with exclusive(args.run/'.lock'):
        result=repack(args.data_root,args.run)
    atomic_json(args.output,result)
    print(json.dumps({k:(len(v) if k=='created_batches' else v) for k,v in result.items()},ensure_ascii=False))
