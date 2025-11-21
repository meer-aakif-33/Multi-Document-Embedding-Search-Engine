from typing import List, Dict, Any
import numpy as np
from ..cache.cache_manager import CacheManager
from ..embedder.embedder import Embedder
from ..indexer.faiss_index import FaissIndex


class SearchEngine:
    def __init__(self, embedder: Embedder, cache: CacheManager, dim: int, index_path: str = "faiss.index"):
        self.embedder = embedder
        self.cache = cache
        self.index = FaissIndex(dim, index_path=index_path)
        self.metadata = {}  # doc_id -> {text, length, filename}

    def index_documents(self, docs: List[Dict]):
        # docs are dicts with doc_id, text, hash, length, filename
        embeddings = []
        doc_ids = []
        for d in docs:
            emb = self.cache.get(d['doc_id'], d['hash'])
            if emb is None:
                # compute and cache
                emb = self.embedder.embed(d['text'])
                self.cache.set(d['doc_id'], d['hash'], emb)
            embeddings.append(emb)
            doc_ids.append(d['doc_id'])
            self.metadata[d['doc_id']] = {
                'text': d['text'],
                'length': d['length'],
                'filename': d['filename']
            }
        if len(embeddings) == 0:
            # nothing to index
            return
        embs = np.vstack(embeddings)
        embs = self.embedder.normalize(embs)
        self.index.build(embs, doc_ids)

    def explain_overlap(self, query: str, doc_text: str) -> Dict[str, Any]:
        # simple tokenizer by whitespace; for better results use a tokenizer or TF-IDF for keywords
        q_words = set(query.lower().split())
        d_words = set(doc_text.lower().split())
        overlap = q_words & d_words
        overlap_count = len(overlap)
        overlap_ratio = overlap_count / max(1, len(q_words))
        # return top overlap keywords (sorted by appearance in doc_text)
        overlap_keywords = []
        if overlap_count > 0:
            # preserve doc order when possible
            seen = set()
            for w in doc_text.lower().split():
                if w in overlap and w not in seen:
                    overlap_keywords.append(w)
                    seen.add(w)
                if len(overlap_keywords) >= 10:
                    break
        return {
            'overlap_keywords': overlap_keywords,
            'overlap_count': overlap_count,
            'overlap_ratio': overlap_ratio
        }

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # embed and normalize query
        q_emb = self.embedder.embed(query)
        q_emb = self.embedder.normalize(q_emb.reshape(1, -1)).squeeze()
        results = self.index.search(q_emb, top_k)
        out = []
        for doc_id, score in results:
            meta = self.metadata.get(doc_id, {})
            explain = self.explain_overlap(query, meta.get('text', ''))
            # length normalization: shorter docs slightly favored (example heuristic)
            length = meta.get('length', 1)
            length_score = 1.0 / (1.0 + (length / 10000.0))
            # combine raw vector score and length normalization into final score
            combined_score = float(score) * 0.8 + length_score * 0.2
            out.append({
                'doc_id': doc_id,
                'score': combined_score,
                'raw_score': float(score),
                'preview': (meta.get('text', '')[:300] + '...') if meta.get('text') else '',
                'explanation': {
                    'keyword_overlap': explain['overlap_keywords'],
                    'overlap_count': explain['overlap_count'],
                    'overlap_ratio': explain['overlap_ratio'],
                    'length_norm': length_score
                },
                'metadata': {
                    'length': length,
                    'filename': meta.get('filename')
                }
            })
        return {'query': query, 'results': out}
