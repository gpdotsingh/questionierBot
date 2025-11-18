# src/splitter.py
from __future__ import annotations
import os, re, json, glob
from typing import Dict, Any, List, Optional

# ----- dotenv (optional) -----
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs): return None

# ----- YAML metadata (optional) -----
try:
    import yaml
except Exception:
    yaml = None

# ----- Optional FAISS vector store wrapper (must provide .load() and .query(text, top_k)) -----
try:
    from src.ingestdata.faiss_store import FaissVectorStoreCosine  # optional
except Exception:
    FaissVectorStoreCosine = None

# ----- LLM providers (optional) -----
try:
    from openai import OpenAI  # OpenAI >= 1.x
except Exception:
    OpenAI = None
try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

# ----- regex helpers -----
JOINERS = re.compile(r"\b(?:and|also|plus|as well as)\b", re.I)
SEQUENCERS = re.compile(r"\b(?:then|next|after|based on|using|with|from)\b", re.I)
SENT_SPLIT = re.compile(r"[.;?!]|\bthen\b", re.I)
CLEAN_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    return CLEAN_WS.sub(" ", s or "").strip()

def _pack_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert list-tree to dict-of-objects:
      [{'id':'Q1','text':'..','children':[{'id':'Q2',...}]}]
      -> {"Q1": {"text":"..","children":[{"Q2":{...}}]}, ...}
    """
    def pack(n: Dict[str, Any]) -> Dict[str, Any]:
        return {n["id"]: {"text": n["text"], "children": [pack(c) for c in n.get("children", [])]}}
    out: Dict[str, Any] = {}
    for n in nodes:
        out.update(pack(n))
    return out

# ---------- tiny provider router ----------
class _LLMRouter:
    """
    Uses .env:
      SPLITTER_PROVIDER=openai|ollama
      OPENAI_API_KEY, OPENAI_MODEL (default gpt-4o)
      OLLAMA_BASE_URL (default http://127.0.0.1:11434), OLLAMA_GEN_MODEL (default llama3.1)
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
        elif self.provider == "ollama" and OllamaClient:
            self._ollama = OllamaClient(host=self.ollama_url)
        else:
            # provider unset or SDK missing -> no LLM path
            self.provider = ""

    def llm_json(self, system: str, user: str) -> Optional[dict]:
        text = self.llm_text(system, user)
        if not text:
            return None
        # robust JSON extraction
        try:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(text[s:e+1])
        except Exception:
            pass
        try:
            return json.loads(text)
        except Exception:
            return None

    def llm_text(self, system: str, user: str) -> Optional[str]:
        try:
            if self._openai:
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
                # If your SDK supports it, you can add: response_format={"type": "json_object"}
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

# ---------- metadata ----------
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

def _known_terms(meta: Dict[str, Any]) -> set:
    terms = set()
    for v in (meta.get("fields") or {}).values():
        terms.add(str(v).lower())
    for k, vs in (meta.get("synonyms") or {}).items():
        terms.add(str(k).lower())
        for s in (vs or []):
            terms.add(str(s).lower())
    return terms

# ---------- vector hints (optional FAISS) ----------
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
    # dedupe; normalize 2-letter tokens to UPPER (e.g., state codes)
    seen, out = set(), []
    for t in bag:
        t2 = t.upper() if (len(t) == 2 and t.isalpha()) else t.lower()
        if t2 not in seen:
            out.append(t2)
            seen.add(t2)
    return out[:20]

# ---------- rule-based fallback (single-root) ----------
def _rule_based_tree(query: str) -> Dict[str, Any]:
    q = _norm(query)
    clauses = [c for c in map(_norm, SENT_SPLIT.split(q)) if c]

    # Virtual root -> children become first-level tasks
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

    # Renumber depth-first & drop virtual root
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

# ---------- LLM path (single-root enforced) ----------
def _llm_plan(user_query: str, meta: Dict[str, Any], router: _LLMRouter) -> Optional[Dict[str, Any]]:
    if not router.provider:
        return None

    schema_lines = []
    if meta.get("fields"):
        schema_lines.append("Fields:")
        for k, v in meta["fields"].items():
            schema_lines.append(f"- {k}: {v}")
    if meta.get("synonyms"):
        schema_lines.append("Synonyms:")
        for k, v in meta["synonyms"].items():
            sv = ", ".join(v) if isinstance(v, list) else str(v)
            schema_lines.append(f"- {k}: {sv}")
    schema_text = "\n".join(schema_lines) if schema_lines else "(none)"

    system = (
        "You split a user’s query into a SINGLE rooted, nested plan. Emit ONLY JSON, no prose.\n"
        "Rules:\n"
        "- Output exactly one root node representing the overall task.\n"
        "- Children that depend on their parent must be inside that parent's \"children\".\n"
        "- Use concise one-line instructions per node.\n"
        "- Do NOT include IDs, only {\"text\",\"children\"}.\n\n"
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
        "}"
    )
    user = (
        f"Metadata:\n{schema_text}\n\n"
        f"User query:\n\"{_norm(user_query)}\"\n"
        f"Return ONLY JSON with a single root in the 'nodes' array."
    )

    raw = router.llm_json(system, user)
    if not raw or not isinstance(raw.get("nodes"), list) or not raw["nodes"]:
        return None

    # Wrap multiple roots under a single root (the user's query)
    if len(raw["nodes"]) > 1:
        raw = {"nodes": [{"text": _norm(user_query), "children": raw["nodes"]}]}

    # Depth-first numbering
    qid = 0
    def renumber(n: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal qid
        qid += 1
        return {
            "id": f"Q{qid}",
            "text": _norm(n.get("text", "")),
            "children": [renumber(c) for c in (n.get("children") or [])],
        }

    numbered_root = [renumber(raw["nodes"][0])]
    return _pack_nodes(numbered_root)

# ---------- public entry ----------
def split_query_simple(user_query: str) -> Dict[str, Any]:
    """
    End-to-end splitter used like:
      from src.splitter import split_query_simple
      print(split_query_simple("Show ..."))
    """
    router = _LLMRouter()
    meta = _load_metadata()

    # 1) Try LLM with metadata awareness
    llm_res = _llm_plan(user_query, meta, router)
    if llm_res:
        return llm_res

    # 2) Otherwise, vector-augmented rule-based split
    aug = _vector_terms(user_query, top_k=8)
    base = f"{user_query} | {' '.join(aug)}" if aug else user_query
    return _rule_based_tree(base)
