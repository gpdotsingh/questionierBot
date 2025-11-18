# src/loaddata.py
from pathlib import Path
from typing import List, Any

# Import loaders with minimal changes. Some versions move classes;
# we keep small try/excepts around specific loaders inside the function.
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
# UnstructuredExcelLoader import can vary; we’ll import lazily in the function.
# JSONLoader signature varies a lot across versions; we’ll also handle it inside the function.


def _load_excel(path_str: str) -> List[Any]:
    """Lazy import for Excel loader with minimal changes across versions."""
    try:
        # Most recent path
        from langchain_community.document_loaders import UnstructuredExcelLoader
    except Exception:
        try:
            # Older path some users had
            from langchain_community.document_loaders.excel import UnstructuredExcelLoader  # type: ignore
        except Exception as e:
            raise ImportError(f"Could not import UnstructuredExcelLoader: {e}")
    loader = UnstructuredExcelLoader(path_str)
    return loader.load()


def _load_json(path_str: str) -> List[Any]:
    """
    JSON loader can be tricky across versions.
    We try LangChain JSONLoader first; if it fails, we fall back to a simple manual loader.
    """
    try:
        from langchain_community.document_loaders import JSONLoader
        # Many versions require jq_schema OR json_lines; we keep it simplest:
        try:
            loader = JSONLoader(file_path=path_str, jq_schema=".", text_content=False)
        except TypeError:
            # Older signatures
            loader = JSONLoader(file_path=path_str)
        return loader.load()
    except Exception:
        # Minimal fallback: read entire file as one document
        import json
        from langchain.docstore.document import Document
        with open(path_str, "r", encoding="utf-8") as f:
            data = f.read()
        return [Document(page_content=data, metadata={"source": path_str, "file_type": "json"})]


def _load_csv(path_str: str) -> List[Any]:
    """
    CSVLoader signatures also differ slightly; keep minimal compatibility.
    """
    try:
        # Newer signature
        loader = CSVLoader(file_path=path_str, csv_args={"delimiter": ","}, encoding="utf-8", autodetect_encoding=True)
    except TypeError:
        # Simpler signatures
        try:
            loader = CSVLoader(file_path=path_str, csv_args={"delimiter": ","}, encoding="utf-8")
        except TypeError:
            loader = CSVLoader(path_str)  # very old versions
    return loader.load()


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents: List[Any] = []

    # PDF
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            for d in loaded:
                d.metadata.setdefault("source_file", pdf_file.name)
                d.metadata.setdefault("file_type", "pdf")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # TXT
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            for d in loaded:
                d.metadata.setdefault("source_file", txt_file.name)
                d.metadata.setdefault("file_type", "txt")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # CSV
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loaded = _load_csv(str(csv_file))
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            # Add useful CSV row metadata if present
            for i, d in enumerate(loaded, start=1):
                d.metadata.setdefault("source_file", csv_file.name)
                d.metadata.setdefault("file_type", "csv")
                d.metadata.setdefault("row", i)
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            loaded = _load_excel(str(xlsx_file))
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
            for d in loaded:
                d.metadata.setdefault("source_file", xlsx_file.name)
                d.metadata.setdefault("file_type", "xlsx")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # Word
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
            for d in loaded:
                d.metadata.setdefault("source_file", docx_file.name)
                d.metadata.setdefault("file_type", "docx")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    # JSON
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loaded = _load_json(str(json_file))
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
            for d in loaded:
                d.metadata.setdefault("source_file", json_file.name)
                d.metadata.setdefault("file_type", "json")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents


# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
