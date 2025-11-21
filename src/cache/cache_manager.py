import sqlite3
import os
import pickle
import time
from typing import Optional
import numpy as np

CACHE_DB = os.environ.get('CACHE_DB', 'embeddings_cache.db')


class CacheManager:
    def __init__(self, db_path: str = CACHE_DB):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cur = self._conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                hash TEXT,
                embedding BLOB,
                updated_at REAL
            )
        ''')
        self._conn.commit()

    def get(self, doc_id: str, hash_val: str) -> Optional[np.ndarray]:
        cur = self._conn.cursor()
        cur.execute('SELECT hash, embedding FROM embeddings WHERE doc_id = ?', (doc_id,))
        row = cur.fetchone()
        if row is None:
            return None
        stored_hash, blob = row
        if stored_hash != hash_val:
            return None
        emb = pickle.loads(blob)
        return emb

    def set(self, doc_id: str, hash_val: str, embedding: np.ndarray):
        cur = self._conn.cursor()
        blob = pickle.dumps(embedding)
        cur.execute('REPLACE INTO embeddings (doc_id, hash, embedding, updated_at) VALUES (?, ?, ?, ?)',
                    (doc_id, hash_val, sqlite3.Binary(blob), time.time()))
        self._conn.commit()

    def all_embeddings(self):
        cur = self._conn.cursor()
        cur.execute('SELECT doc_id, embedding FROM embeddings')
        rows = cur.fetchall()
        result = {}
        for doc_id, blob in rows:
            result[doc_id] = pickle.loads(blob)
        return result

    def close(self):
        self._conn.close()


if __name__ == '__main__':
    # small test
    import numpy as np
    cm = CacheManager(':memory:')
    cm.set('doc1', 'h1', np.array([1.0, 2.0]))
    print(cm.get('doc1', 'h1'))