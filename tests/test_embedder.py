from src.embedder.embedder import Embedder

def test_embedder_single():
    emb = Embedder()
    vec = emb.embed("hello world")
    assert vec.shape[0] > 0
