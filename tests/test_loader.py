# tests/test_loader.py
import os
from src.document_loader.loader import load_documents

def test_load_documents_basic():
    docs = load_documents('data/docs')
    assert isinstance(docs, list)
    assert len(docs) > 0
    first = docs[0]
    assert 'doc_id' in first
    assert 'text' in first
    assert 'hash' in first
    assert 'length' in first
    assert 'filename' in first

# tests/test_embedder.py
from src.embedder.embedder import Embedder

def test_embedder_single():
    emb = Embedder()
    vec = emb.embed("hello world")
    assert vec.shape[0] > 0

# tests/test_cache.py
import numpy as np
from src.cache.cache_manager import CacheManager

def test_cache_set_get():
    cm = CacheManager(':memory:')
    arr = np.array([1.0,2.0,3.0])
    cm.set('doc1','h1',arr)
    out = cm.get('doc1','h1')
    assert out.tolist() == arr.tolist()

# tests/test_search.py
from src.embedder.embedder import Embedder
from src.cache.cache_manager import CacheManager
from src.retriever.search_engine import SearchEngine

def test_search_engine_small():
    embedder = Embedder()
    cache = CacheManager(':memory:')
    dim = embedder.embed("test").shape[0]
    engine = SearchEngine(embedder, cache, dim)
    docs = [
        {'doc_id':'d1','text':'machine learning basics','hash':'h1','length':25,'filename':'x'},
        {'doc_id':'d2','text':'cooking pasta recipe','hash':'h2','length':23,'filename':'y'}
    ]
    engine.index_documents(docs)
    res = engine.search('machine learning', top_k=1)
    assert res['results'][0]['doc_id'] == 'd1'
