# scripts/build_vectors.py
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestdata.ingest_pipeline import run_ingestion

if __name__ == "__main__":
    print("[DEBUG] sys.path[0:5] =", sys.path[:5])
    run_ingestion(
        data_dir="data",
        yaml_path="metadata/crm_donor_data.yaml",
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
    )