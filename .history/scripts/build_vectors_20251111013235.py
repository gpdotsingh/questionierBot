# scripts/build_vectors.py
from src.ingest_pipeline import run_ingestion

if __name__ == "__main__":
    run_ingestion(
        data_dir="data",
        yaml_path="metadata/crm_donor_data.yaml",
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
    )
