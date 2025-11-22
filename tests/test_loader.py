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
