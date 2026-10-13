"""Content-addressed rows for explicitly batch-independent embedding models."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .storage import IntegrityError, canonical, digest


def row_id(model, text):
    return digest({"base_url":model["base_url"],"model":model["model"],"input":text})


class EmbeddingRows:
    def __init__(self, path: Path):
        self.path=path
        path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=60)
        # Use the ordinary rollback journal: the data volume can be a network FS.
        self.db.execute("CREATE TABLE IF NOT EXISTS rows (id TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)")
        self.db.commit()

    def close(self):
        self.db.close()

    def get(self, model, texts):
        identities={row_id(model,t):t for t in texts}
        if not identities:return {}
        result={}
        keys=list(identities)
        for offset in range(0,len(keys),200):
            batch=keys[offset:offset+200]
            sql="SELECT id,payload,sha256 FROM rows WHERE id IN ("+",".join("?" for _ in batch)+")"
            for key,raw,checksum in self.db.execute(sql,batch):
                value=json.loads(raw)
                if digest(value)!=checksum or value['input_hash']!=digest(identities[key]):
                    raise IntegrityError('embedding row content hash mismatch')
                result[identities[key]]=value
        return result

    def put(self, model, texts, vectors, provenance):
        if len(texts)!=len(vectors):raise ValueError('embedding row count mismatch')
        with self.db:
            for text,vector in zip(texts,vectors):
                key=row_id(model,text)
                value={'input_hash':digest(text),'vector':vector,'provenance':provenance}
                row=self.db.execute('SELECT payload,sha256 FROM rows WHERE id=?',(key,)).fetchone()
                if row:
                    old=json.loads(row[0])
                    if digest(old)!=row[1] or old['input_hash']!=digest(text):
                        raise IntegrityError('embedding row content hash mismatch')
                    if old['vector']!=vector:
                        raise IntegrityError('conflicting vectors under the same text/model identity; pin a distinct model profile')
                else:
                    self.db.execute('INSERT INTO rows VALUES(?,?,?)',(key,canonical(value),digest(value)))
