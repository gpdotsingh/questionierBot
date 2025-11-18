python
# filepath: questionierBot/src/qapipeline/splitter.py
from __future__ import annotations
import os, re, json, glob
from typing import Dict, Any, List, Optional
from .models import Step, Plan

# ---------- dotenv ----------
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs): return None

# ---------- YAML metadata ----------
try:
    import yaml
except Exception:
    yaml = None

# ---------- Optional FAISS store (cosine) ----------
try:
    from src.ingestdata.faiss_store import FaissVectorStoreCosine
except Exception:
    FaissVectorStoreCosine = None

# ---------- LLM providers ----------
try:
    from openai import OpenAI  # >=1.x
except Exception:
    OpenAI = None

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
    def pack(n: Dict[str, Any]) -> Dict[str, Any]:
        return {n["id"]: {"text": n["text"], "children": [pack(c) for c in n.get("children", [])]}}
    out: Dict[str, Any] = {}
    for n in nodes:
        out.update(pack(n))
    return out

class _LLMRouter:
    def __init__(self) -> None:
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
        else:
            self.provider = ""

    @staticmethod
    def _prompt_header() -> str:
        return (
            "You are a planner that splits a user question into Q-steps.\n"
            "Emit ONLY JSON in this EXACT shape, no prose:\n"
            "{\n"
            "  \"Q1\": {\"text\": \"...\", \"children\": [ {\"Q2\": {\"text\": \"...\", \"children\": []}}, {\"Q3\": {\"text\": \"...\", \"children\": []}} ]},\n"
            "  \"Q4\": {\"text\": \"...\", \"children\": []}\n"
            "}\n"
            "Rules:\n"
            "- Use brief, executable texts per node.\n"
            "- If a sub-question depends on its parent, nest it under parent's children.\n"
            "- You may create multiple roots (Q1, Q4, ...), or a single root with deep children.\n"
            "- NEVER add commentary.\n"
            "- Do not include IDs inside texts. IDs are Q1..Qn only.\n"
        )

    @staticmethod
    def _prompt_body(user_query: str, meta_text: str, hint_text: str = "") -> str:
        block_meta = f"Metadata:\n{meta_text}\n" if meta_text else "Metadata:\n(none)\n"
        block_hint = f"\nVectorHints:\n{hint_text}\n" if hint_text else ""
        return (
            f"{block_meta}"
            f"UserQuery:\n\"{_norm(user_query)}\"\n"
            f"{block_hint}\n"
            "Output JSON now:"
        )

    def ask_json(self, prompt: str) -> Optional[dict]:
        try:
            if self._openai:
                resp = self._openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                text = resp.choices[0].message.content or ""
            elif self._ollama:
                resp = self._ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1},
                )
                text = (resp.get("message") or {}).get("content") or str(resp)
            else:
                return None
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(text[s:e+1])
            return json.loads(text)
        except Exception:
            return None

def _load_metadata(dirs: Optional[List[str]] = None) -> Dict[str, Any]:
    dirs = dirs or ["metadata", "metadat"]
    merged: Dict[str, Any] = {"fields": {}, "synonyms": {}}
    for d in dirs:
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

def _metadata_text(meta: Dict[str, Any]) -> str:
    if not meta:
        return ""
    lines: List[str] = []
    if meta.get("fields"):
        lines.append("Fields:")
        for k, v in meta["fields"].items():
            lines.append(f"- {k}: {v}")
    if meta.get("synonyms"):
        lines.append("Synonyms:")
        for k, vs in meta["synonyms"].items():
            if isinstance(vs, list):
                vs = ", ".join(vs)
            lines.append(f"- {k}: {vs}")
    return "\n".join(lines)

def _vector_terms(query: str, top_k: int = 8, faiss_dir: str = "faiss_store") -> List[str]:
    if FaissVectorStoreCosine is None:
        return []
    try:
        store = FaissVectorStoreCosine(persist_dir=faiss_dir)
        store.load()
        hits = store.query(query, top_k=top_k) or []
    except Exception:
        return []
    bag: List[str] = []
    for h in hits:
        md = (h.get("metadata") or {})
        for _, val in md.items():
            if isinstance(val, str):
                bag.extend(re.findall(r"[A-Za-z0-9_]{2,}", val))
    out, seen = [], set()
    for t in bag:
        t2 = t.upper() if (len(t) == 2 and t.isalpha()) else t.lower()
        if t2 not in seen:
            seen.add(t2); out.append(t2)
    return out[:20]

def _rule_based_tree(query: str) -> Dict[str, Any]:
    q = _norm(query)
    clauses = [c for c in map(_norm, SENT_SPLIT.split(q)) if c]
    root = {"id": "QROOT", "text": q, "children": []}
    cursor = root
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
        first = new_node(parts[0])
        cursor["children"].append(first)
        parent_for_next = first
        for p in parts[1:]:
            node = new_node(p)
            if sequenced:
                parent_for_next["children"].append(node)
                parent_for_next = node
            else:
                cursor["children"].append(node)
        if sequenced:
            cursor = parent_for_next
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
        real_roots = [{"id": "Q1", "text": q, "children": []}]
    return _pack_nodes(real_roots)

def _number_from_llm_dict(llm_dict: dict) -> Dict[str, Any]:
    if "Q1" in llm_dict or any(k.startswith("Q") for k in llm_dict.keys()):
        return llm_dict
    nodes = llm_dict.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("LLM output not in expected shape")
    qid = 0
    def renumber(n: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal qid
        qid += 1
        return {
            "id": f"Q{qid}",
            "text": _norm(n.get("text", "")),
            "children": [renumber(c) for c in (n.get("children") or [])],
        }
    numbered = [renumber(n) for n in nodes]
    return _pack_nodes(numbered)

class QuestionSplitter:
    """
    Wraps original functionality:
      - Loads metadata
      - Optionally queries LLM (OpenAI/Ollama) with prompt-first strategy
      - Fallback to rule-based decomposition
      - plan(question) -> Plan (flattened steps)
    Public signature preserved: __init__(try_llm=True, provider=None, model=None)
    provider/model are ignored (env-driven).
    """
    def __init__(
        self,
        try_llm: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        faiss_dir: str = "faiss_store",
        metadata_dirs: Optional[List[str]] = None
    ):
        self.try_llm = try_llm
        self.faiss_dir = faiss_dir
        self.metadata_dirs = metadata_dirs or ["metadata", "metadat"]
        self.router = _LLMRouter() if try_llm else None
        self.meta_cache = _load_metadata(self.metadata_dirs)
        self.meta_text = _metadata_text(self.meta_cache)

    def _attempt_llm(self, user_query: str) -> Optional[Dict[str, Any]]:
        if not (self.try_llm and self.router and self.router.provider):
            return None
        prompt = _LLMRouter._prompt_header() + "\n" + _LLMRouter._prompt_body(user_query, self.meta_text)
        raw = self.router.ask_json(prompt)
        if isinstance(raw, dict):
            try:
                return _number_from_llm_dict(raw)
            except Exception:
                pass
        hints = " ".join(self._vector_terms(user_query))
        if hints:
            prompt2 = _LLMRouter._prompt_header() + "\n" + _LLMRouter._prompt_body(user_query, self.meta_text, hint_text=hints)
            raw2 = self.router.ask_json(prompt2)
            if isinstance(raw2, dict):
                try:
                    return _number_from_llm_dict(raw2)
                except Exception:
                    pass
        return None

    def _vector_terms(self, query: str) -> List[str]:
        return _vector_terms(query, top_k=8, faiss_dir=self.faiss_dir)

    def _fallback_tree(self, query: str) -> Dict[str, Any]:
        augmented = query
        if not (self.router and self.router.provider):
            terms = self._vector_terms(query)
            if terms:
                augmented = f"{query} | {' '.join(terms)}"
        return _rule_based_tree(augmented)

    def _flatten_tree(self, tree: Dict[str, Any]) -> List[Step]:
        steps: List[Step] = []
        def walk(node_key: str, node_val: Dict[str, Any]):
            steps.append(Step(node_key, node_val.get("text", "")))
            for child in node_val.get("children", []):
                for ck, cv in child.items():
                    walk(ck, cv)
        for k, v in sorted(tree.items(), key=lambda kv: int(kv[0][1:]) if kv[0].startswith("Q") and kv[0][1:].isdigit() else 0):
            walk(k, v)
        return steps

    def plan(self, question: str) -> Plan:
        q = _norm(question)
        tree = self._attempt_llm(q)
        used_llm = tree is not None
        if tree is None:
            tree = self._fallback_tree(q)
        steps = self._flatten_tree(tree)
        if not steps:
            steps = [Step("Q1", q)]
        return Plan(original_question=q, used_llm=used_llm, ordered_steps=steps)

def split_query_simple(user_query: str) -> Dict[str, Any]:
    splitter = QuestionSplitter()
    tree = splitter._attempt_llm(user_query) or splitter._fallback_tree(user_query)
    return tree