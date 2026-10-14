import pytest
import numpy as np
from store import FaissVectorStore

@pytest.fixture
def temp_faiss_store(tmp_path):
    index_path = str(tmp_path / "test_faiss.bin")
    meta_path = str(tmp_path / "test_meta.pkl")
    store = FaissVectorStore(dimension=512, index_path=index_path, meta_path=meta_path)
    return store
