# src/ingestdata/ingest_pipeline.py
from typing import Optional
from .data_loader import load_all_documents
from .yaml_loader import load_yaml_semantic_documents
from .faiss_store import FaissVectorStoreCosine

def run_ingestion(
    data_dir: str = "data",
    yaml_path: str = "metadata/quickbooks_data.yaml",
    persist_dir: str = "faiss_store",
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    semantics_yaml_path: str = "data/quickbooks_semantics.yaml",
):
    # Load CSV documents (existing)
    csv_docs = load_all_documents(data_dir)

    # Load YAML semantic documents (new)
    yaml_docs = load_yaml_semantic_documents(
        semantics_yaml_path=semantics_yaml_path,
        data_yaml_path=yaml_path,
    )

    # Combine both document sets
    all_docs = csv_docs + yaml_docs
    print(f"[PIPELINE] Total documents: {len(csv_docs)} CSV + {len(yaml_docs)} YAML = {len(all_docs)}")

    store = FaissVectorStoreCosine(
        persist_dir=persist_dir,
        yaml_path=yaml_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    store.build_from_documents(all_docs)
    print("[PIPELINE] Ingestion completed.")
