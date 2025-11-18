# ingest_vectors.py
# Create a Chroma vector index from a donor CSV using LangChain Core docs.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain_core.documents import Document

# Prefer modern packages
try:
    from langchain_chroma import Chroma
except Exception:
    # Fallback for older installs
    from langchain_community.vectorstores import Chroma  # type: ignore

# Embeddings: choose hf (default) or ollama
def get_embeddings(kind: str = "hf"):
    kind = (kind or "hf").lower().strip()
    if kind == "ollama":
        from langchain_ollama import OllamaEmbeddings
        # Uses whatever model you have in Ollama, e.g., "nomic-embed-text" or "all-minilm"
        # Pull an embedding model once via: `ollama pull nomic-embed-text`
        return OllamaEmbeddings(model="nomic-embed-text")
    else:
        # HuggingFace default: small, fast, solid quality
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            # Fallback community path
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def normalize_date(value: Any) -> Optional[str]:
    """Parse dates robustly and return ISO-8601 string or None."""
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value, errors="coerce").date().isoformat()
    except Exception:
        return None


def coerce_float(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def coerce_int(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        f = coerce_float(value)
        return int(f) if f is not None else None


def build_document(row: pd.Series, idx: int, source: str) -> Document:
    """
    Convert one donor row into a LangChain Document.
    - page_content: short natural-language blurb for semantic search.
    - metadata: rich, typed fields so you can filter later.
    """
    # Column names expected in your dataset (common aliases handled below).
    # Prefer your actual headers: DonorID, FirstName, LastName, Email, Phone, City, State,
    # ZipCode, LastDonationDate, TotalGifts, TotalAmountDonated, EventParticipation, EngagementScore
    donor_id = str(row.get("DonorID", "")).strip()
    first = str(row.get("FirstName", "")).strip()
    last = str(row.get("LastName", "")).strip()
    email = str(row.get("Email", "")).strip()
    phone = str(row.get("Phone", "")).strip()
    city = str(row.get("City", "")).strip()
    state = str(row.get("State", "")).strip().upper()
    zipcode = str(row.get("ZipCode", "")).strip()

    last_dt = normalize_date(row.get("LastDonationDate", None))
    total_gifts = coerce_int(row.get("TotalGifts", None))
    total_amount = coerce_float(row.get("TotalAmountDonated", None))
    engagement = coerce_int(row.get("EngagementScore", None))

    # Normalize boolean-ish event participation
    ev_raw = str(row.get("EventParticipation", "")).strip().lower()
    if ev_raw in ("yes", "true", "y", "1"):
        event_participation = True
    elif ev_raw in ("no", "false", "n", "0", ""):
        event_participation = False
    else:
        event_participation = None

    # A compact sentence that captures the row’s semantics
    page_content = (
        f"Donor {donor_id} ({first} {last}) from {city}, {state} "
        f"last donated on {last_dt or 'unknown date'}; "
        f"total gifts {total_gifts if total_gifts is not None else 'NA'}, "
        f"total amount {total_amount if total_amount is not None else 'NA'}, "
        f"engagement score {engagement if engagement is not None else 'NA'}."
    )

    metadata: Dict[str, Any] = {
        # identity & contact
        "donor_id": donor_id,
        "first_name": first or None,
        "last_name": last or None,
        "email": email or None,
        "phone": phone or None,
        # geography
        "city": city or None,
        "state": state or None,          # keep uppercase for easy 2-letter filters
        "zip_code": zipcode or None,
        # donation stats
        "last_donation_date": last_dt,
        "total_gifts": total_gifts,
        "total_amount_donated": total_amount,
        "engagement_score": engagement,
        "event_participation": event_participation,
        # provenance
        "source": str(source),
        "row": idx,                       # original row index
        "schema_version": "donor_csv.v1"  # handy if you evolve your metadata
    }

    return Document(page_content=page_content, metadata=metadata)


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Optional: harmonize known aliases (only if your file uses variants)
    alias_map = {
        "Donor Id": "DonorID",
        "Donor_Id": "DonorID",
        "GiftCount": "TotalGifts",
        "Gifts": "TotalGifts",
        "Amount": "TotalAmountDonated",
        "DonationAmount": "TotalAmountDonated",
        "Date": "LastDonationDate",
        "GiftDate": "LastDonationDate",
        "Engagement": "EngagementScore",
        "Score": "EngagementScore",
        "Participated": "EventParticipation",
        "Event": "EventParticipation",
        "Zip": "ZipCode",
    }
    cols = {c: alias_map.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    return df


def main():
    ap = argparse.ArgumentParser(description="Ingest donor CSV into a Chroma vector store.")
    ap.add_argument("--csv", required=True, help="Path to CSV file (e.g., data/CRM_Donor_Simulation_Dataset.csv)")
    ap.add_argument("--persist", default="vectors", help="Chroma persist directory (default: ./vectors)")
    ap.add_argument("--collection", default="donors", help="Chroma collection name (default: donors)")
    ap.add_argument("--embeddings", choices=["hf", "ollama"], default="hf", help="Embedding backend: hf or ollama")
    ap.add_argument("--recreate", action="store_true", help="Delete existing collection data before ingest")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    persist_dir = Path(args.persist).expanduser().resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] CSV: {csv_path}")
    print(f"[ingest] Persist dir: {persist_dir}")
    print(f"[ingest] Collection: {args.collection}")
    print(f"[ingest] Embeddings: {args.embeddings}")

    df = load_csv(csv_path)
    print(f"[ingest] Rows: {len(df)}; Columns: {list(df.columns)}")

    # Build documents
    docs: List[Document] = []
    for i, row in df.iterrows():
        docs.append(build_document(row, idx=int(i), source=str(csv_path)))

    # Embeddings
    embeddings = get_embeddings(args.embeddings)

    # Optionally clear existing data for a fresh build
    if args.recreate:
        # Chroma py-client doesn’t have a simple “drop collection” through langchain wrapper,
        # so easiest is to remove the directory.
        import shutil
        print("[ingest] Recreate requested: removing existing persist directory...")
        if persist_dir.exists():
            shutil.rmtree(persist_dir)

    # Create (or load) the vector store and add documents
    # Note: passing collection_name ensures predictable naming
    vs = Chroma(
        collection_name=args.collection,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    print("[ingest] Adding documents to Chroma...")
    # add_documents returns IDs; not strictly needed here
    _ = vs.add_documents(docs)
    vs.persist()

    print("[ingest] Done. Vector store is ready.")
    print(f"[ingest] Try a quick similarity search later, e.g.:")
    print(f"         \"Which donors in CA have engagement > 80?\" (filter with where={{'state': 'CA'}} on the metadata)")
    print(f"         or programmatically via LangChain retrievers.")


if __name__ == "__main__":
    main()
