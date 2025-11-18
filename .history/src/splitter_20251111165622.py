from __future__ import annotations
import os, re, json, glob
from typing import Dict, Any, List, Optional
from pathlib import Path

# --- minimal, resilient imports ---
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs): return None

try:
    import yaml
except Exception:
    yaml = None

try:
    import pandas as pd
except Exception:
    pd = None

# OpenAI (>=1.x) optional
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

# Ollama optional
try:
    from ollama import Client as _OllamaClient
except Exception:
    _OllamaClient = None


# ---------------- utilities ----------------
_JOINERS = re.compile(r"\b(?:and|also|as well as|plus|together with)\b", re.I)
_SEQUENCERS = re.compile(r"\b(?:then|after|after that|next|based on|using|with|from)\b", re.I)
_SENT_SPLIT = re.compile(r"[.;?!]|\bthen\b", re.I)
_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())

def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


# ---------------- LLM router (very small) ----------------
class LLM:
    def __init__(self):
        load_dotenv()
        self.provider = (os.getenv("SPLITTER_PROVIDER") or "").strip().lower()
        # auto-pick if not set
        if not self.provider:
            self.provider = "openai" if os.getenv("OPENAI_API_KEY") else ("ollama" if os.getenv("OLLAMA_BASE_URL") else "")
        self.oai_key = os.getenv("OPENAI_API_KEY")
        self.oai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_GEN_MODEL", "llama3.1")

        self._oai = _OpenAI(api_key=self.oai_key) if (self.provider == "openai" and _OpenAI and self.oai_key) else None
        self._ollama = _OllamaClient(host=self.ollama_url) if (self.provider == "ollama" and _OllamaClient) else None

    def json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        txt = self.text(system, user)
        if not txt:
            return None
        # tolerant JSON extraction
        try:
            i, j = txt.find("{"), txt.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(txt[i:j+1])
            return json.loads(txt)
        except Exception:
            return None

    def text(self, system: str, user: str) -> Optional[str]:
        try:
            if self._oai:
                resp = self._oai.chat.completions.create(
                    model=self.oai_model,
                    temperature=0.1,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                )
                return resp.choices[0].message.content
            if self._ollama:
                full = f"System: {system}\n\nUser: {user}"
                resp = self._ollama.chat(model=self.ollama_model, messages=[{"role":"user","content":full}], options={"temperature":0.1})
                return (resp.get("message") or {}).get("content") or str(resp)
        except Exception:
            return None
        return None


# ---------------- metadata loader ----------------
def load_all_metadata(dirpath: str = "metadata") -> Dict[str, Any]:
    merged: Dict[str, Any] = {"fields":{}, "synonyms":{}}
    if not yaml:
        return merged
    for p in glob.glob(os.path.join(dirpath, "*.yaml")) + glob.glob(os.path.join(dirpath, "*.yml")):
        try:
            with open(p, "r") as f:
                y = yaml.safe_load(f) or {}
            # shallow merge for fields/synonyms
            for k in ("fields", "synonyms"):
                if k in y and isinstance(y[k], dict):
                    merged[k].update(y[k])
        except Exception:
            pass
    return merged


# ---------------- Step 3: LLM split using metadata ----------------
def try_llm_split_with_metadata(llm: LLM, user_query: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fields = meta.get("fields", {})
    synonyms = meta.get("synonyms", {})
    system = (
        "You decompose the user's question into a nested plan for data analysis.\n"
        "Output VALID JSON ONLY in this exact shape:\n"
        '{ "nodes": [ { "text": "root task", "children": [ { "text": "child", "children": [] } ] } ] }\n'
        "If the user asks multiple independent tasks, produce multiple root nodes.\n"
        "Children represent sub-queries that depend on their parent result."
    )
    user = f"""Metadata (use for terms only):
Fields: {json.dumps(fields)}
Synonyms: {json.dumps(synonyms)}

Question: "{user_query}"

Return ONLY JSON as described."""
    plan = llm.json(system, user) if llm else None
    if not plan or "nodes" not in plan:
        return None
    return plan


# ---------------- Step 4: DB/CSV mining fallback ----------------
def mine_terms_from_data(user_query: str, data_dir: str = "data", max_rows: int = 2000) -> List[str]:
    """
    Very simple, fast fallback:
    - scans CSVs in ./data
    - picks columns whose name intersects tokens from the query
    - collects a few distinct values as hints (states, ids, etc.)
    """
    if not pd:
        return []
    toks = set(re.findall(r"[A-Za-z0-9_]+", user_query.lower()))
    hints: List[str] = []
    for csvp in Path(data_dir).glob("*.csv"):
        try:
            df = pd.read_csv(csvp, nrows=max_rows)
        except Exception:
            continue
        for col in df.columns:
            cl = str(col).lower()
            if any(t in cl for t in toks):
                # harvest up to 10 sample values as strings
                vals = df[col].dropna().astype(str).head(10).tolist()
                hints.extend(vals)
        # also harvest likely standard columns if present
        for k in ["State","DonorID","City","ZipCode","LastDonationDate","TotalGifts","TotalAmountDonated","EngagementScore"]:
            if k in df.columns:
                vals = df[k].dropna().astype(str).head(10).tolist()
                hints.extend(vals)
    return _dedup_keep_order(hints)[:30]


# ---------------- splitting (rules) ----------------
def rule_split_to_nodes(text: str) -> List[Dict[str, Any]]:
    """
    Siblings: separated by 'and/also/plus...'  -> Q1, Q2, Q3
    Nested:   if clause uses 'then/after/using...' -> becomes children of previous node
    """
    clauses = [c for c in map(_norm, _SENT_SPLIT.split(text)) if c]
    nodes: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []
    qid = 0

    def new_node(t: str) -> Dict[str, Any]:
        nonlocal qid
        qid += 1
        return {"id": f"Q{qid}", "text": t, "children": []}

    for clause in clauses:
        parts = [p for p in map(_norm, _JOINERS.split(clause)) if p]
        is_seq = bool(_SEQUENCERS.search(clause))
        for i, part in enumerate(parts):
            node = new_node(part)
            if is_seq or i > 0:
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


# ---------------- final shape ----------------
def nodes_to_Qshape(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Emit JSON shaped like:
      {
        "Q1": {"text":"...", "children":[ {"Q2":{...}}, {"Q3":{...}} ]},
        "Q4": {"text":"...", "children":[]}
      }
    """
    def pack(n: Dict[str, Any]) -> Dict[str, Any]:
        return { n["id"]: { "text": n["text"], "children": [pack(c) for c in n.get("children", [])] } }
    out: Dict[str, Any] = {}
    for n in nodes:
        out.update(pack(n))
    return out


# ---------------- orchestrated splitter (your 5 steps) ----------------
def split_query(user_query: str, metadata_dir: str = "metadata", data_dir: str = "data") -> Dict[str, Any]:
    """
    Steps:
      1) Load the env.
      2) Use LLM from the .env file.
      3) Fetch metadata and try LLM split.
      4) If LLM+metadata fails: mine terms from data, augment query, rule-split.
      5) Return required JSON Q-shape.
    """
    # 1) env + 2) LLM
    llm = LLM()

    # 3) metadata-driven LLM split
    meta = load_all_metadata(metadata_dir)
    plan = try_llm_split_with_metadata(llm, user_query, meta)
    if plan and isinstance(plan.get("nodes"), list) and plan["nodes"]:
        # number them Q1.. and return
        numbered = _number_nodes(plan["nodes"])
        return nodes_to_Qshape(numbered)

    # 4) DB/CSV mining fallback -> augment then rule split
    mined_terms = mine_terms_from_data(user_query, data_dir=data_dir)
    augmented = (user_query + " | " + " ".join(mined_terms)) if mined_terms else user_query
    nodes = rule_split_to_nodes(augmented)

    # 5) final Q-shape
    return nodes_to_Qshape(nodes)


def _number_nodes(raw_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign Q1/Q2... IDs depth-first to agent JSON: [{"text": "...", "children":[...]}]"""
    out: List[Dict[str, Any]] = []
    qid = 0
    def attach(node: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal qid
        qid += 1
        me = {"id": f"Q{qid}", "text": _norm(node.get("text","")), "children": []}
        for c in node.get("children") or []:
            me["children"].append(attach(c))
        return me
    for n in raw_nodes:
        out.append(attach(n))
    return out


# -------- tiny CLI for quick test --------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "show donors by state then compute median per-donor totals; also list top 5 states"
    result = split_query(q, metadata_dir="metadata", data_dir="data")
    print(json.dumps(result, indent=2))
