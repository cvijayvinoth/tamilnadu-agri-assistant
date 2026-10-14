import numpy as np

def test_faiss_store_add_and_search(temp_faiss_store):
    dummy_vector = np.random.rand(1, 512).astype(np.float32)
    meta = [{"text": "Drip irrigation scheme", "source": "GO_123.pdf", "page": 1}]
    
    temp_faiss_store.add_vectors(dummy_vector, meta)
    results = temp_faiss_store.search(dummy_vector, k=1)
    
    assert len(results) == 1
    assert results[0]["text"] == "Drip irrigation scheme"
    assert results[0]["source"] == "GO_123.pdf"
