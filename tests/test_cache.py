import numpy as np
from src.cache.cache_manager import CacheManager

def test_cache_set_get():
    cm = CacheManager(':memory:')
    arr = np.array([1.0,2.0,3.0])
    cm.set('doc1','h1',arr)
    out = cm.get('doc1','h1')
    assert out.tolist() == arr.tolist()
