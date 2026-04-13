#!/usr/bin/env python3
"""
Build a separate FAISS vector DB for report metadata.

Ingests two YAML files:
  - metadata/quickbooks_reports.yaml   (report catalog — query_sequence, params)
  - metadata/report_decision_guide.yaml (decision guide — use_when, do_not_use_when)

Produces:  faiss_reports_store/  (faiss.index + metadata.pkl)

Usage:
    python ingest_reports.py
    python ingest_reports.py --persist-dir faiss_reports_store

Query examples (in Python):
    from src.ingestdata.faiss_store import FaissVectorStoreCosine

    store = FaissVectorStoreCosine(persist_dir="faiss_reports_store")
    store.load()

    # Decision guide only (splitter uses this)
    hits = store.query("revenue by customer", k=5,
                       filter={"source": "report_decision_guide"})

    # Report catalog only (orchestrator uses this)
    hits = store.query("revenue by customer", k=5,
                       filter={"source": "report_catalog"})

    # Both sources (no filter)
    hits = store.query("revenue by customer", k=5)
"""
import argparse

from src.ingestdata.yaml_loader import (
    load_report_catalog_documents,
    load_report_guide_documents,
)
from src.ingestdata.faiss_store import FaissVectorStoreCosine


def main():
    parser = argparse.ArgumentParser(description="Ingest report YAMLs into FAISS")
    parser.add_argument(
        "--catalog",
        default="metadata/quickbooks_reports.yaml",
        help="Path to the report catalog YAML",
    )
    parser.add_argument(
        "--guide",
        default="metadata/report_decision_guide.yaml",
        help="Path to the report decision guide YAML",
    )
    parser.add_argument(
        "--persist-dir",
        default="faiss_reports_store",
        help="Directory for the FAISS index + metadata",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model",
    )
    args = parser.parse_args()

    catalog_docs = load_report_catalog_documents(args.catalog)
    guide_docs = load_report_guide_documents(args.guide)

    all_docs = catalog_docs + guide_docs
    print(f"[INGEST_REPORTS] Total: {len(catalog_docs)} catalog + {len(guide_docs)} guide = {len(all_docs)} documents")

    store = FaissVectorStoreCosine(
        persist_dir=args.persist_dir,
        yaml_path=None,
        embedding_model=args.model,
        chunk_size=2000,
        chunk_overlap=100,
    )
    store.build_from_documents(all_docs)
    print(f"[INGEST_REPORTS] Done. Index saved to {args.persist_dir}/")


if __name__ == "__main__":
    main()
