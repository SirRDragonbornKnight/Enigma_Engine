"""
Very small placeholder vector utilities. For real vector DB, use FAISS, Annoy, Milvus, or Weaviate.
We include a simple in-memory store for demo.
"""
import numpy as np

class SimpleVectorDB:
    def __init__(self, dim=128):
        self.dim = dim
        self.vectors = []
        self.ids = []

    def add(self, vec, id_):
        arr = np.asarray(vec).astype(float)
        if arr.shape[0] != self.dim:
            raise ValueError("Expected dim", self.dim)
        self.vectors.append(arr)
        self.ids.append(id_)

    def search(self, qvec, topk=5):
        if not self.vectors:
            return []
        d = np.stack(self.vectors, axis=0)
        q = np.asarray(qvec).astype(float)
        dots = d @ q
        idx = np.argsort(-dots)[:topk]
        return [(self.ids[i], float(dots[i])) for i in idx]
