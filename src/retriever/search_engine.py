#search_engine
from typing import List, Dict, Any
import numpy as np

from src.utils.hashing import sha256_text
from ..cache.cache_manager import CacheManager
from ..embedder.embedder import Embedder
from ..indexer.faiss_index import FaissIndex
from ..embedder.batch_embedder import embed_batch_multiprocess

class SearchEngine:
    def __init__(self, embedder: Embedder, cache: CacheManager, dim: int, index_path: str = "faiss.index"):
        self.embedder = embedder
        self.cache = cache
        self.index = FaissIndex(dim, index_path=index_path)
        self.metadata = {}  # doc_id -> {text, length, filename}

    def index_documents(self, docs: List[Dict]):
        """
        Build embeddings for all docs:
        - Use cache when possible
        - Use batch multiprocessing for uncached docs
        - Always store metadata
        """
        texts_to_embed = []
        ids_to_embed = []
        hashes_to_embed = []

        embeddings = []
        doc_ids = []

        # STEP 1 — Collect cached and uncached docs
        for d in docs:

            # Always store metadata
            self.metadata[d['doc_id']] = {
                'text': d['text'],
                'length': d['length'],
                'filename': d['filename']
            }

            cached = self.cache.get(d['doc_id'], d['hash'])
            if cached is not None:
                embeddings.append(cached)
                doc_ids.append(d['doc_id'])
            else:
                texts_to_embed.append(d['text'])
                ids_to_embed.append(d['doc_id'])
                hashes_to_embed.append(d['hash'])  # correct hash

        # STEP 2 — Batch embed all uncached docs
        if texts_to_embed:
            batch_embs = embed_batch_multiprocess(texts_to_embed)
            for doc_id, emb, h in zip(ids_to_embed, batch_embs, hashes_to_embed):
                # store in cache using correct text hash
                self.cache.set(doc_id, h, emb)
                embeddings.append(emb)
                doc_ids.append(doc_id)

        # Nothing to index?
        if not embeddings:
            return

        # STEP 3 — Build FAISS index
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
