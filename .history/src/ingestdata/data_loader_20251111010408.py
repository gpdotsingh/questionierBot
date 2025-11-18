# src/ingestdata/data_loader.py
from pathlib import Path
from typing import List
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document

def load_all_documents(data_dir: str) -> List[Document]:
    """
    Load ALL CSV files in data_dir into LangChain Documents (one doc per row).
    Adds row index & filename as metadata for traceability.
    """
    p = Path(data_dir)
    docs: List[Document] = []
    for csv_path in p.rglob("*.csv"):
        try:
            loader = CSVLoader(
                file_path=str(csv_path),
                csv_args={"delimiter": ","},
                encoding="utf-8",
                autodetect_encoding=True,
            )
        except TypeError:
            loader = CSVLoader(file_path=str(csv_path), csv_args={"delimiter": ","}, encoding="utf-8")

        rows = loader.load()
        for i, d in enumerate(rows, start=1):
            md = dict(d.metadata)
            md["source_file"] = csv_path.name
            md["row"] = i
            md["file_type"] = "csv"
            docs.append(Document(page_content=d.page_content, metadata=md))

        print(f"[LOAD] {csv_path.name}: {len(rows)} rows")

    print(f"[LOAD] Total docs: {len(docs)}")
    return docs
