import os
import pickle
import numpy as np
import faiss
from config import EMBEDDING_DIM, FAISS_INDEX_PATH, METADATA_STORE_PATH

class FaissVectorStore:
    def __init__(self, dimension: int = EMBEDDING_DIM, index_path: str = FAISS_INDEX_PATH, meta_path: str = METADATA_STORE_PATH):
        self.dimension = dimension
        self.index_path = index_path
        self.meta_path = meta_path
        self.metadata_store = {}

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata_store = pickle.load(f)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))

    def add_vectors(self, vectors: np.ndarray, metadata_list: list[dict]):
        vectors = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        
        start_id = len(self.metadata_store)
        ids = np.arange(start_id, start_id + len(vectors)).astype(np.int64)

        self.index.add_with_ids(vectors, ids)
        for idx, meta in zip(ids, metadata_list):
            self.metadata_store[idx] = meta

        self.save()

    def search(self, query_vector: np.ndarray, k: int = 2) -> list[dict]:
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        if len(indices) > 0:
            for idx, score in zip(indices[0], distances[0]):
                if idx != -1 and idx in self.metadata_store:
                    item = self.metadata_store[idx].copy()
                    item["score"] = float(score)
                    results.append(item)
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata_store, f)
