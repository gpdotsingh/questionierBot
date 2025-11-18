# src/qapipeline/agentic_splitter.py
from __future__ import annotations
import os, re, json, glob
from typing import Dict, Any, List, Optional

# ---------- dotenv ----------
try:
    from dotenv import load_dotenv
except Exception:  # fallback if dotenv not installed
    def load_dotenv(*args, **kwargs): return None

# ---------- YAML (metadata) ----------
try:
    import yaml
except Exception:
    yaml = None

# ---------- Optional FAISS vector store (cosine wrapper) ----------
# Expect class with .load() and .query(text, top_k) -> [{"metadata": {...}}, ...]
try:
    from src.ingestdata.faiss_store import FaissVectorStoreCosine  # your cosine impl
except Exception:
    FaissVectorStoreCosine = None

# ---------- LLM providers ----------
# OpenAI (>=1.x SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Ollama
try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

# ---------- Regex helpers ----------
JOINERS = re.compile(r"\b(?:and|also|plus|as well as)\b", re.I)
SEQUENCERS = re.compile(r"\b(?:then|next|after|based on|using|with|from)\b", re.I)
SENT_SPLIT = re.compile(r"[.;?!]|\bthen\b", re.I)
CLEAN_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    return CLEAN_WS.sub(" ", s or "").strip()

def _pack_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Turn a numbered tree list like:
      [{'id':'Q1','text':'..','children':[{'id':'Q2',...}]}]
    into the requested dict-of-objects:
      { "Q1": {"text":"..","children":[{"Q2":{...}}]} }
    """
    def pack(n: Dict[str, Any]) -> Dict[str, Any]:
        return {n["id"]: {"text": n["text"], "children": [pack(c) for c in n.get("children", [])]}}
    out: Dict[str, Any] = {}
    for n in nodes:
        out.update(pack(n))
    return out


class _LLMRouter:
    """
    Minimal provider router. Uses:
      SPLITTER_PROVIDER=openai|ollama
      OPENAI_API_KEY, OPENAI_MODEL   (default gpt-4o)
      OLLAMA_BASE_URL, OLLAMA_GEN_MODEL (default llama3.1)
    """
    def __init__(self) -> None:
        load_dotenv()
        self.provider = (os.getenv("SPLITTER_PROVIDER") or "").lower().strip()
        # OpenAI
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        # Ollama
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_GEN_MODEL", "llama3.1")

        self._openai = None
        self._ollama = None
        if self.provider == "openai" and OpenAI and self.openai_key:
            self._openai = OpenAI(api_key=self.openai_key)
            print("[LLMRouter] Using OpenAI:", self.openai_model)
        elif self.provider == "ollama" and OllamaClient:
            self._ollama = OllamaClient(host=self.ollama_url)
            print("[LLMRouter] Using Ollama:", self.ollama_model)
        else:
            if self.provider:
                print("[LLMRouter] Provider configured but SDK not available; falling back to rules.")

    def llm_json(self, system: str, user: str) -> Optional[dict]:
        txt = self.llm_text(system, user)
        if not txt:
            return None
        # Try to extract JSON robustly
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

    def llm_text(self, system: str, user: str) -> Optional[str]:
        try:
            if self._openai:
                # If your SDK supports it, you can add response_format={"type":"json_object"}
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
                resp = self._openai.chat.completions.create(
                    model=self.openai_model, messages=msgs, temperature=0.1
                )
                return resp.choices[0].message.content
            elif self._ollama:
                full = f"System: {system}\n\nUser: {user}"
                resp = self._ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": full}],
                    options={"temperature": 0.1},
                )
                return (resp.get("message") or {}).get("content") or str(resp)
            return None
        except Exception:
            return None


class AgenticSplitter:
    """
    End-to-end, environment-driven splitter:
      1) Loads env for provider selection
      2) Reads YAML metadata (fields + synonyms) from metadata/ or metadat/
      3) Tries LLM nested plan (single root)
      4) Falls back to rule-based parser, optionally augmented by FAISS vector hints
      5) Emits strictly numbered, single-root JSON of shape:
         {
           "Q1": {"text":"...", "children":[ {"Q2":{...}}, {"Q3":{...}} ]}
         }
    """

    def __init__(self,
                 metadata_dirs: Optional[List[str]] = None,
                 faiss_dir: str = "faiss_store"):
        self.metadata_dirs = metadata_dirs or ["metadata", "metadat"]
        self.faiss_dir = faiss_dir
        self.llm = _LLMRouter()
        self.meta = self._load_metadata()
        self._terms = self._known_terms_from_meta(self.meta)

    # --------- public API ---------
    def split(self, user_query: str) -> Dict[str, Any]:
        user_query = _norm(user_query)

        # 1) Try LLM with metadata awareness
        plan = self._llm_nested_plan(user_query)
        if plan is not None:
            return plan

        # 2) Otherwise try vector-augmented rule-based split
        aug_terms = self._vector_terms(user_query, top_k=8)
        base = f"{user_query} | {' '.join(aug_terms)}" if aug_terms else user_query
        return self._rule_based_tree(base)

    # --------- metadata ----------
    def _load_metadata(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {"fields": {}, "synonyms": {}}
        for d in self.metadata_dirs:
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

    def _known_terms_from_meta(self, meta: Dict[str, Any]) -> set:
        terms = set()
        for v in (meta.get("fields") or {}).values():
            terms.add(str(v).lower())
        for k, vs in (meta.get("synonyms") or {}).items():
            terms.add(str(k).lower())
            for s in (vs or []):
                terms.add(str(s).lower())
        return terms

    # --------- LLM path ----------
    def _llm_nested_plan(self, user_query: str) -> Optional[Dict[str, Any]]:
        if not self.llm.provider:
            return None

        schema_lines = []
        if self.meta.get("fields"):
            schema_lines.append("Fields:")
            for k, v in self.meta["fields"].items():
                schema_lines.append(f"- {k}: {v}")
        if self.meta.get("synonyms"):
            schema_lines.append("Synonyms:")
            for k, v in self.meta["synonyms"].items():
                sv = ", ".join(v) if isinstance(v, list) else str(v)
                schema_lines.append(f"- {k}: {sv}")
        schema_text = "\n".join(schema_lines) if schema_lines else "(none)"

        system = (
            "You split a user’s query into a SINGLE rooted, nested plan. Emit ONLY JSON, no prose.\n"
            "Rules:\n"
            "- Output exactly one root node that represents the overall task.\n"
            "- Every child that depends on its parent must be inside the parent's \"children\".\n"
            "- Do not create siblings when a “then / after / based on / using / with” relation exists; make them children instead.\n"
            "- Do not include IDs. Only provide \"text\" and \"children\".\n"
            "- Keep it concise; one sentence max per node.\n\n"
            "Schema (exact):\n"
            "{\n"
            "  \"nodes\": [\n"
            "    { \"text\": \"root task\", \"children\": [\n"
            "        { \"text\": \"child 1\", \"children\": [] },\n"
            "        { \"text\": \"child 2\", \"children\": [\n"
            "            { \"text\": \"grandchild\", \"children\": [] }\n"
            "        ]}\n"
            "    ]}\n"
            "  ]\n"
            "}\n"
            "User must see ONLY valid JSON."
        )

        prompt = (
            f"Use the metadata to decompose the user's query into the nested plan.\n\n"
            f"Metadata:\n{schema_text}\n\n"
            f"User query:\n\"{user_query}\"\n"
        )

        raw = self.llm.llm_json(system, prompt)
        if not raw or not isinstance(raw.get("nodes"), list) or not raw["nodes"]:
            return None

        # If model returned multiple roots, wrap them under a single virtual root of the user's query
        if len(raw["nodes"]) > 1:
            raw = {"nodes": [{"text": user_query, "children": raw["nodes"]}]}

        # Depth-first numbering to guarantee unique, ordered Q-ids
        qid = 0
        def renumber(n: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal qid
            qid += 1
            return {
                "id": f"Q{qid}",
                "text": _norm(n.get("text", "")),
                "children": [renumber(c) for c in (n.get("children") or [])]
            }

        numbered_roots = [renumber(raw["nodes"][0])]
        return _pack_nodes(numbered_roots)

    # --------- vector augmentation (FAISS) ----------
    def _vector_terms(self, query: str, top_k: int = 8) -> List[str]:
        if FaissVectorStoreCosine is None:
            return []
        try:
            store = FaissVectorStoreCosine(persist_dir=self.faiss_dir)
            store.load()
            hits = store.query(query, top_k=top_k) or []
        except Exception:
            return []
        bag: List[str] = []
        for h in hits:
            md = (h.get("metadata") or {})
            for key, val in (md.items()):
                if isinstance(val, str):
                    # collect tokens (states, fields, etc.)
                    bag.extend(re.findall(r"[A-Za-z0-9_]{2,}", val))
        # de-dup while preserving order (normalize 2-letter tokens uppercase)
        seen, out = set(), []
        for t in bag:
            t2 = t.upper() if (len(t) == 2 and t.isalpha()) else t.lower()
            if t2 not in seen:
                out.append(t2)
                seen.add(t2)
        return out[:20]

    # --------- rule-based fallback (single-root) ----------
    def _rule_based_tree(self, query: str) -> Dict[str, Any]:
        q = _norm(query)
        clauses = [c for c in map(_norm, SENT_SPLIT.split(q)) if c]

        # Create a single virtual root (the raw query); children become first-level tasks
        root = {"id": "QROOT", "text": q, "children": []}
        current_parent = root

        qid = 0
        def new_node(text: str) -> Dict[str, Any]:
            nonlocal qid
            qid += 1
            return {"id": f"Q{qid}", "text": text, "children": []}

        for clause in clauses:
            parts = [p for p in map(_norm, JOINERS.split(clause)) if p]
            if not parts:
                continue
            sequenced = bool(SEQUENCERS.search(clause))

            # First part is a child of the current parent
            first = new_node(parts[0])
            current_parent["children"].append(first)

            # Subsequent parts: if sequenced, chain as children; else, siblings under current_parent
            parent_for_next = first
            for p in parts[1:]:
                node = new_node(p)
                if sequenced:
                    parent_for_next["children"].append(node)
                    parent_for_next = node
                else:
                    current_parent["children"].append(node)

            # If the clause indicates sequence, push the cursor down
            if sequenced:
                current_parent = parent_for_next

        # Renumber depth-first & drop the virtual root (its children become the actual roots)
        qid2 = 0
        def walk(n: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal qid2
            qid2 += 1
            return {
                "id": f"Q{qid2}",
                "text": n["text"],
                "children": [walk(c) for c in n.get("children", [])]
            }

        real_roots = [walk(c) for c in root.get("children", [])]
        if not real_roots:
            # If we couldn't split, return the whole query as single root
            real_roots = [{"id": "Q1", "text": q, "children": []}]
        return _pack_nodes(real_roots)
