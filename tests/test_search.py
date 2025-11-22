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
