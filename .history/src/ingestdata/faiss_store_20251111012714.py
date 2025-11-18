# src/ingestdata/faiss_store.py
import os, pickle, yaml
import faiss
import numpy as np
from typing import List, Any, Dict, Optional

from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from .embedding import EmbeddingPipeline

def _unit_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

class FaissVectorStoreCosine:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        yaml_path: Optional[str] = "metadata/crm_donor_data.yaml",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.yaml_path = yaml_path
        self.schema = self._load_yaml(yaml_path)

        self.index = None
        self.metadata: List[Dict[str, Any]] = []

        self.embedding_model = embedding_model
        self.query_model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -------- YAML ----------
    def _load_yaml(self, path: Optional[str]) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _map_meta(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        fields = (self.schema.get("fields") or {})
        ds = (self.schema.get("dataset") or {})
        meta = (self.schema.get("meta") or {})
        out = {
            "dataset": ds.get("name", "unknown"),
            "granularity": ds.get("granularity", "unknown"),
            "source_type": meta.get("source_type", raw.get("file_type")),
            "source_path": meta.get("source_path", raw.get("source_file")),
        }
        def pick(key: str):
            src = fields.get(key)
            if not src: return None
            return raw.get(src) or raw.get(src.lower()) or raw.get(key)
        out.update({
            "donor_id": pick("donor_id"),
            "first_name": pick("first_name"),
            "last_name": pick("last_name"),
            "email": pick("email"),
            "phone": pick("phone"),
            "city": pick("city"),
            "state": pick("state"),
            "zipcode": pick("zipcode"),
            "last_donation_date": pick("last_donation_date"),
            "total_gifts": pick("total_gifts"),
            "total_amount": pick("total_amount"),
            "event_participation": pick("event_participation"),
            "engagement_score": pick("engagement_score"),
        })
        out["_raw"] = dict(raw)
        return out

    # -------- Build from docs ----------
    def build_from_documents(self, documents: List[Document]) -> None:
        pipe = EmbeddingPipeline(self.embedding_model, self.chunk_size, self.chunk_overlap)
        chunks = pipe.chunk_documents(documents)
        embs = pipe.embed_chunks(chunks)
        vecs = _unit_norm(np.asarray(embs, dtype="float32"))

        metas: List[Dict[str, Any]] = []
        for i, c in enumerate(chunks):
            m = self._map_meta(c.metadata)
            m["text"] = c.page_content
            m["chunk_index"] = i
            m["content_length"] = len(c.page_content)
            metas.append(m)

        self._add(vecs, metas)
        self.save()

    def _add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        dim = vecs.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # cosine via IP on unit vectors
        self.index.add(vecs)
        self.metadata.extend(metas)
        print(f"[FAISS] Added {vecs.shape[0]} vectors (dim={dim}). Total={self.index.ntotal}")

    # -------- Persist --------
    def save(self) -> None:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "embedding_model": self.embedding_model,
                "yaml_path": self.yaml_path
            }, f)
        print(f"[FAISS] Saved to {self.persist_dir}")

    def load(self) -> None:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            saved = pickle.load(f)
        self.metadata = saved["metadata"]
        self.embedding_model = saved.get("embedding_model", self.embedding_model)
        self.yaml_path = saved.get("yaml_path", self.yaml_path)
        self.query_model = SentenceTransformer(self.embedding_model)
        print(f"[FAISS] Loaded {self.index.ntotal} vectors.")

    # -------- Query --------
    def query(self, text: str, k: Optional[int] = None, min_score: float = 0.1):
        if self.index is None or self.index.ntotal == 0:
            return []
        if k is None:
            k = self.index.ntotal  # return ALL by default

        q = self.query_model.encode([text]).astype("float32")
        q = _unit_norm(q)
        scores, idxs = self.index.search(q, k)  # IP in [-1,1]
        out = []
        for i, s in zip(idxs[0], scores[0]):
            if i < 0: continue
            if s < min_score: continue
            meta = self.metadata[i] if i < len(self.metadata) else {}
            out.append({"index": int(i), "score": float(s), "metadata": meta})
        return out
