# scripts/query_vectors.py
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestdata.faiss_store import FaissVectorStoreCosine

if __name__ == "__main__":
    store = FaissVectorStoreCosine(persist_dir="faiss_store")
    store.load()
    q = "donors from AK with high engagement"
    results = store.query(q)  # ALL matches by default
    print(f"Query: {q}")
    print(f"Got {len(results)} results; top 10:")
    for r in results[:5]:
        md = r["metadata"]
        print(f"  score={r['score']:.4f} donor={md.get('donor_id')} state={md.get('state')} text_snip={md.get('text','')[:60]!r}")
