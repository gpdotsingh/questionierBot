from __future__ import annotations
import os, re, json, glob
from typing import Dict, Any, List, Optional

# ---------- dotenv ----------
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs): return None

# ---------- YAML (metadata) ----------
try:
    import yaml
except Exception:
    yaml = None

# ---------- Optional FAISS vector store (your cosine wrapper) ----------
try:
    from src.ingestdata.faiss_store import FaissVectorStoreCosine  # must provide .load() and .query(text, top_k)
except Exception:
    FaissVectorStoreCosine = None

# ---------- LLM providers ----------
# OpenAI (>=1.x)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Ollama
try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

# ---------- tiny utils ----------
JOINERS = re.compile(r"\b(?:and|also|plus|as well as)\b", re.I)
SEQUENCERS = re.compile(r"\b(?:then|next|after|based on|using|with|from)\b", re.I)
SENT_SPLIT = re.compile(r"[.;?!]|\bthen\b", re.I)
CLEAN_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    return CLEAN_WS.sub(" ", s or "").strip()

def _pack_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn [{'id':'Q1','text':..., 'children':[...]}] into the requested JSON shape."""
    def pack(n: Dict[str, Any]) -> Dict[str, Any]:
        return {n["id"]: {"text": n["text"], "children": [pack(c) for c in n.get("children", [])]}}
    out: Dict[str, Any] = {}
    for n in nodes:
        out.update(pack(n))
    return out

# ---------- LLM Router (very small) ----------
class LLMRouter:
    def __init__(self):
        load_dotenv()
        self.provider = (os.getenv("SPLITTER_PROVIDER") or "").lower().strip()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_GEN_MODEL", "llama3.1")

        self._openai = None
        self._ollama = None

        if self.provider == "openai" and OpenAI and self.openai_key:
            self._openai = OpenAI(api_key=self.openai_key)
        elif self.provider == "ollama" and OllamaClient:
            self._ollama = OllamaClient(host=self.ollama_url)

    def llm_json(self, prompt: str, system: Optional[str] = None) -> Optional[dict]:
        txt = self.llm_text(prompt, system)
        if not txt:
            return None
        # extract JSON robustly
        try:
            s, e = txt.find("{"), txt.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(txt[s:e+1])
        except Exception:
            pass
        try:
            return json.loads(txt)
        except Exception:
            return None

    def llm_text(self, prompt: str, system: Optional[str] = None) -> Optional[str]:
        try:
            if self._openai:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = self._openai.chat.completions.create(
                    model=self.openai_model, messages=messages, temperature=0.1
                )
                return resp.choices[0].message.content
            elif self._ollama:
                full = f"System: {system}\n\nUser: {prompt}" if system else prompt
                resp = self._ollama.chat(model=self.ollama_model, messages=[{"role": "user", "content": full}], options={"temperature": 0.1})
                return (resp.get("message") or {}).get("content") or str(resp)
            else:
                return None
        except Exception:
            return None

# ---------- Metadata loader ----------
def load_metadata_dir() -> Dict[str, Any]:
    """
    Reads all *.yaml/*.yml files from ./metadata or ./metadat.
    Merges keys: fields, synonyms (simple).
    """
    base_dirs = ["metadata", "metadat"]
    merged: Dict[str, Any] = {"fields": {}, "synonyms": {}}
    for d in base_dirs:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.y*ml")):
            if not yaml:
                continue
            try:
                with open(p, "r") as f:
                    y = yaml.safe_load(f) or {}
                for k in ("fields", "synonyms"):
                    if k in y and isinstance(y[k], dict):
                        merged[k].update(y[k])
            except Exception:
                pass
    return merged

def known_terms_from_meta(meta: Dict[str, Any]) -> set:
    terms = set()
    for v in (meta.get("fields") or {}).values():
        terms.add(str(v).lower())
    for k, vs in (meta.get("synonyms") or {}).items():
        terms.add(str(k).lower())
        for s in (vs or []):
            terms.add(str(s).lower())
    return terms

# ---------- very small rule-based splitter ----------
def rule_split(query: str) -> List[Dict[str, Any]]:
    q = _norm(query)
    clauses = [c for c in map(_norm, SENT_SPLIT.split(q)) if c]
    nodes: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []
    qid = 0

    def new_node(text: str) -> Dict[str, Any]:
        nonlocal qid
        qid += 1
        return {"id": f"Q{qid}", "text": text, "children": []}

    for clause in clauses:
        parts = [p for p in map(_norm, JOINERS.split(clause)) if p]
        seq = bool(SEQUENCERS.search(clause))
        for i, part in enumerate(parts):
            node = new_node(part)
            if seq or i > 0:
                parent = stack[-1] if stack else (nodes[-1] if nodes else None)
                if parent is None:
                    nodes.append(node)
                else:
                    parent["children"].append(node)
                    stack.append(node)
            else:
                nodes.append(node)
                stack = [node]
    return nodes

# ---------- vector backfill terms ----------
def vector_terms(query: str, top_k: int = 8) -> List[str]:
    if FaissVectorStoreCosine is None:
        return []
    try:
        store = FaissVectorStoreCosine(persist_dir="faiss_store")
        store.load()
        hits = store.query(query, top_k=top_k) or []
    except Exception:
        return []
    bag: List[str] = []
    for h in hits:
        md = (h.get("metadata") or {})
        for key in md.keys():
            val = md.get(key)
            if isinstance(val, str):
                bag.extend(re.findall(r"[A-Za-z0-9_]{2,}", val))
    # de-dup, keep order
    seen, out = set(), []
    for t in bag:
        t2 = t.upper() if (len(t) == 2 and t.isalpha()) else t.lower()
        if t2 not in seen:
            out.append(t2)
            seen.add(t2)
    return out[:20]

# ---------- main entry ----------
def split_query_simple(user_query: str) -> Dict[str, Any]:
    """
    1) Load env
    2) Use LLM from .env
    3) Fetch metadata and try LLM split with it
    4) If that fails, use vector DB to augment and rule-split
    5) Return JSON in requested structure
    """
    load_dotenv()
    llm = LLMRouter()
    meta = load_metadata_dir()
    terms = known_terms_from_meta(meta)

    # Try LLM with metadata
    schema_text = ""
    if meta.get("fields"):
        schema_text += "Fields:\n" + "\n".join(f"- {k}: {v}" for k, v in meta["fields"].items())
    if meta.get("synonyms"):
        schema_text += "\nSynonyms:\n" + "\n".join(f"- {k}: {', '.join(v)}" for k, v in meta["synonyms"].items())

    if llm.provider:
        system = "You split user questions into a nested plan. Output ONLY JSON as specified."
        prompt = f"""
Use the metadata to decompose the user's query into a nested plan.
Return ONLY JSON in this shape:
{{
  "nodes":[
    {{"text":"root task 1","children":[{{"text":"child","children":[]}}]}},
    {{"text":"root task 2","children":[]}}
  ]
}}

Metadata:
{schema_text or "(none)"}

User query: "{_norm(user_query)}"
"""
        plan = llm.llm_json(prompt, system)
        if plan and isinstance(plan.get("nodes"), list) and plan["nodes"]:
            # Number nodes as Q1... and pack
            numbered: List[Dict[str, Any]] = []
            qid = 0

            def attach(n: Dict[str, Any]) -> Dict[str, Any]:
                nonlocal qid
                qid += 1
                kids = [attach(k) for k in (n.get("children") or [])]
                return {"id": f"Q{qid}", "text": _norm(n.get("text", "")), "children": kids}

            for root in plan["nodes"]:
                numbered.append(attach(root))
            return _pack_nodes(numbered)

    # If LLM path unavailable/failed: use vector to harvest hints, then rule split
    aug = vector_terms(user_query)
    base = user_query
    if aug:
        base = f"{user_query} | {' '.join(aug)}"

    nodes = rule_split(base)
    return _pack_nodes(nodes)
