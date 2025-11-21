# faiss_index.py (with persistence)
import numpy as np
import os
from typing import List, Tuple

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


class FaissIndex:
    def __init__(self, dim: int, index_path: str = "faiss.index"):
        self.dim = dim
        self.index_path = index_path
        self.doc_ids = []
        self.index = None

        # Load index if exists
        if _FAISS_AVAILABLE and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"[INFO] Loaded FAISS index from {self.index_path}")
        elif _FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

    def build(self, embeddings: np.ndarray, doc_ids: List[str]):
        if _FAISS_AVAILABLE:
            if self.index.ntotal == 0:
                # Build from scratch
                self.index.add(embeddings.astype('float32'))
            else:
                # Replace existing index
                self.index = faiss.IndexFlatIP(self.dim)
                self.index.add(embeddings.astype('float32'))
        else:
            self._fallback_embs = embeddings
        self.doc_ids = list(doc_ids)

        # Save updated index
        if _FAISS_AVAILABLE:
            faiss.write_index(self.index, self.index_path)
            print(f"[INFO] Saved FAISS index to {self.index_path}")

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        q = query_emb.reshape(1, -1).astype('float32')

        if _FAISS_AVAILABLE:
            scores, idxs = self.index.search(q, top_k)
            scores = scores[0].tolist()
            idxs = idxs[0].tolist()
            results = []
            for s, i in zip(scores, idxs):
                if i < 0:
                    continue
                results.append((self.doc_ids[i], float(s)))
            return results
        else:
            dots = (self._fallback_embs @ q.T).squeeze()
            order = np.argsort(-dots)[:top_k]
            return [(self.doc_ids[i], float(dots[i])) for i in order]


# main.py modification snippet
"""
from ..indexer.faiss_index import FaissIndex

@app.on_event("startup")
def startup():
    global ENGINE
    data_folder = os.environ.get("DATA_FOLDER", "data/docs")
    docs = load_documents(data_folder)
    cache = CacheManager(os.environ.get("CACHE_DB", "embeddings_cache.db"))
    embedder = Embedder()
    dim = embedder.embed("test").shape[0]

    ENGINE = SearchEngine(embedder, cache, dim, index_path="faiss.index")
    ENGINE.index_documents(docs)
"""
