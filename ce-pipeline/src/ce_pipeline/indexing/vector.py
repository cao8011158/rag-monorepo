from __future__ import annotations
import numpy as np
import faiss

def build_faiss_index(emb: np.ndarray, index_type: str = "FlatIP") -> faiss.Index:
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    d = emb.shape[1]
    if index_type == "FlatIP":
        index = faiss.IndexFlatIP(d)
    elif index_type == "FlatL2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"Unsupported faiss index_type: {index_type}")
    index.add(emb)
    return index
