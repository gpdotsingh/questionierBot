# src/ingestdata/yaml_loader.py
"""
Converts quickbooks_semantics.yaml (+ quickbooks_data.yaml for types/valid_values)
into LangChain Document objects — one document per entity — for FAISS ingestion.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from langchain_core.documents import Document


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def _build_entity_text(
    entity_name: str,
    sem_info: Dict[str, Any],
    data_info: Optional[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> str:
    """Build a rich text description of an entity for embedding."""
    lines: List[str] = []
    lines.append(f"Entity: {entity_name}")

    # Fields with types
    fields = sem_info.get("fields", [])
    types = (data_info or {}).get("types", {})
    if fields:
        field_parts = []
        for f in fields:
            ftype = types.get(f, "")
            field_parts.append(f"{f} ({ftype})" if ftype else f)
        lines.append(f"Fields: {', '.join(field_parts)}")

    # Valid values
    valid_values = (data_info or {}).get("valid_values", {})
    if valid_values:
        vv_parts = []
        for col, vals in valid_values.items():
            if isinstance(vals, list):
                vv_parts.append(f"{col}: {vals}")
        if vv_parts:
            lines.append(f"Valid Values: {'; '.join(vv_parts)}")

    # Synonyms (from semantics entity)
    synonyms = sem_info.get("synonyms", {})
    if synonyms:
        syn_parts = []
        for key, vals in synonyms.items():
            if isinstance(vals, list):
                syn_parts.append(f"{key} = [{', '.join(vals)}]")
            else:
                syn_parts.append(f"{key} = {vals}")
        lines.append(f"Synonyms: {'; '.join(syn_parts)}")

    # Dimensions
    dims = sem_info.get("dimensions", [])
    if dims:
        lines.append(f"Dimensions: {', '.join(dims)}")

    # Measures
    measures = sem_info.get("measures", [])
    if measures:
        lines.append(f"Measures: {', '.join(measures)}")

    # Relationships
    entity_rels = [r for r in relationships if r.get("from") == entity_name or r.get("to") == entity_name]
    if entity_rels:
        rel_parts = []
        for r in entity_rels:
            rel_parts.append(f"{r['from']} -> {r['to']} (via {r['via']}, {r.get('type', '')})")
        lines.append(f"Relationships: {'; '.join(rel_parts)}")

    return "\n".join(lines)


def load_yaml_semantic_documents(
    semantics_yaml_path: str = "data/quickbooks_semantics.yaml",
    data_yaml_path: str = "metadata/quickbooks_data.yaml",
) -> List[Document]:
    """
    Load the semantics YAML and produce one Document per entity.
    Merges type/valid_value info from data_yaml_path when available.
    """
    sem = _load_yaml(semantics_yaml_path)
    data = _load_yaml(data_yaml_path)

    sem_entities = sem.get("entities", {})
    data_entities = data.get("entities", {})
    relationships = sem.get("relationships", [])

    docs: List[Document] = []
    for entity_name, sem_info in sem_entities.items():
        if not isinstance(sem_info, dict):
            continue
        data_info = data_entities.get(entity_name)
        text = _build_entity_text(entity_name, sem_info, data_info, relationships)
        fields_list = sem_info.get("fields", [])
        doc = Document(
            page_content=text,
            metadata={
                "entity_name": entity_name,
                "fields": ", ".join(fields_list) if isinstance(fields_list, list) else str(fields_list),
                "source_file": "quickbooks_semantics.yaml",
                "file_type": "yaml",
            },
        )
        docs.append(doc)

    # Also add global synonyms as a separate document for search
    global_synonyms = sem.get("synonyms", {})
    if global_synonyms:
        syn_lines = ["Global Synonyms for QuickBooks Entities:"]
        for key, vals in global_synonyms.items():
            if isinstance(vals, list):
                syn_lines.append(f"  {key}: {', '.join(vals)}")
        syn_doc = Document(
            page_content="\n".join(syn_lines),
            metadata={
                "entity_name": "_global_synonyms",
                "source_file": "quickbooks_semantics.yaml",
                "file_type": "yaml",
            },
        )
        docs.append(syn_doc)

    # Add relationships as a separate document
    if relationships:
        rel_lines = ["Entity Relationships:"]
        for r in relationships:
            rel_lines.append(f"  {r['from']} -> {r['to']} via {r['via']} ({r.get('type', '')})")
        rel_doc = Document(
            page_content="\n".join(rel_lines),
            metadata={
                "entity_name": "_relationships",
                "source_file": "quickbooks_semantics.yaml",
                "file_type": "yaml",
            },
        )
        docs.append(rel_doc)

    print(f"[YAML_LOADER] Loaded {len(docs)} documents from {semantics_yaml_path}")
    return docs
