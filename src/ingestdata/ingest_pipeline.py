# src/ingestdata/ingest_pipeline.py
from typing import Optional
from .data_loader import load_all_documents
from .faiss_store import FaissVectorStoreCosine

def run_ingestion(
    data_dir: str = "data",
    yaml_path: str = "metadata/crm_donor_data.yaml",
    persist_dir: str = "faiss_store",
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    docs = load_all_documents(data_dir)
    store = FaissVectorStoreCosine(
        persist_dir=persist_dir,
        yaml_path=yaml_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    store.build_from_documents(docs)
    print("[PIPELINE] Ingestion completed.")
