# src/faiss_store.py
import os
import faiss
import numpy as np
import pickle
import yaml
from typing import List, Any, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# Reuse your EmbeddingPipeline (unchanged)
from src.ingestdata.embedding import EmbeddingPipeline


def _to_unit(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors so IP == cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


class FaissVectorStoreCosine:
    """
    FAISS-backed vector store using cosine similarity (via inner product on unit vectors).
    - Reads a YAML to project CSV/LangChain Document metadata into a consistent schema.
    - Returns ALL matches by default (unless user passes k).
    """

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
        self.schema: Dict[str, Any] = self._load_schema(yaml_path)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Track if vectors in the index are already normalized (they should be)
        self._uses_cosine = True

        print(f"[INFO] Loaded embedding model: {embedding_model}")
        if self.schema:
            print(f"[INFO] Loaded YAML schema: {yaml_path}")

    # ---------- YAML ----------
    def _load_schema(self, yaml_path: Optional[str]) -> Dict[str, Any]:
        if not yaml_path or not os.path.exists(yaml_path):
            return {}
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _project_metadata(self, doc_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw Document.metadata to a clean, schema-driven metadata dict."""
        fields = (self.schema.get("fields") or {})
        mapped = {
            "dataset": (self.schema.get("dataset") or {}).get("name", "unknown"),
            "granularity": (self.schema.get("dataset") or {}).get("granularity", "unknown"),
            "source_type": (self.schema.get("meta") or {}).get("source_type", "unknown"),
            "source_path": (self.schema.get("meta") or {}).get("source_path", "unknown"),
        }

        def pick(key: str) -> Optional[Any]:
            src = fields.get(key)
            if not src:
                return None
            # support both CSV row dicts and LangChain loader metadata keys
            return doc_meta.get(src) or doc_meta.get(src.lower()) or doc_meta.get(key) or None

        mapped.update({
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
        # Keep raw metadata keys as a sub-object for traceability
        mapped["_raw"] = dict(doc_meta)
        return mapped

    # ---------- Build from documents ----------
    def build_from_documents(self, documents: List[Any]) -> None:
        """
        Chunk -> embed -> normalize -> index -> persist.
        Documents are LangChain-style documents with .page_content and .metadata.
        """
        print(f"[INFO] Building FAISS (cosine) from {len(documents)} documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)  # (N, d) float64 or float32

        # Normalize for cosine
        vecs = _to_unit(np.asarray(embeddings, dtype="float32"))

        # Prepare metadata aligned 1:1 with vecs
        metadatas: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            m = self._project_metadata(chunk.metadata)
            m["text"] = chunk.page_content
            m["chunk_index"] = i
            m["content_length"] = len(chunk.page_content)
            metadatas.append(m)

        self._add_embeddings(vecs, metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    # ---------- Low-level add/search/persist ----------
    def _add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        dim = embeddings.shape[1]
        if self.index is None:
            # Cosine via IP on unit vectors
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors (dim={dim}) to FAISS index.")

    def save(self) -> None:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "embedding_model": self.embedding_model,
                "uses_cosine": self._uses_cosine,
                "yaml_path": self.yaml_path,
            }, f)
        print(f"[INFO] Saved FAISS index + metadata to: {self.persist_dir}")

    def load(self) -> None:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            saved = pickle.load(f)
        self.metadata = saved["metadata"]
        self.embedding_model = saved.get("embedding_model", self.embedding_model)
        self._uses_cosine = saved.get("uses_cosine", True)
        self.yaml_path = saved.get("yaml_path", self.yaml_path)
        print(f"[INFO] Loaded FAISS index ({self.index.ntotal} vectors) from: {self.persist_dir}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search the index with a precomputed query embedding.
        - By default, returns ALL results sorted by score (cosine/IP), filtered by min_score.
        - If k is provided, returns top-k.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if k is None:
            k = self.index.ntotal  # return all by default

        # Normalize query for cosine/IP
        q = _to_unit(query_embedding.astype("float32"))
        scores, indices = self.index.search(q, k)  # IP scores in [-1,1], higher is better

        out: List[Dict[str, Any]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            if score < min_score:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            out.append({
                "index": int(idx),
                "score": float(score),
                "metadata": meta,
            })
        return out

    def query(self, query_text: str, k: Optional[int] = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Text → embed → search. Returns all matches unless k is set."""
        print(f"[INFO] Query: {query_text}")
        q_emb = self.model.encode([query_text])
        q_emb = np.asarray(q_emb, dtype="float32")
        return self.search(q_emb, k=k, min_score=min_score)
