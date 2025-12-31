from __future__ import annotations
from typing import List, Optional
from .models import Plan
import os
import re

# ---------- Optional FAISS store (cosine) ----------
try:
    from src.ingestdata.faiss_store import FaissVectorStoreCosine
except Exception:
    FaissVectorStoreCosine = None

class Orchestrator:
    """
    Dummy orchestrator: for each step produces a synthetic answer string.
    Input: Plan
    Output: List[str] (answers) passed to compiler.
    """
    def __init__(self, debug: bool = False, **kwargs):
        self.debug = debug

    def run(self, plan: Plan) -> List[str]:
        self.provider = (os.getenv("ORCHESTRATOR_PROVIDER") or "").lower().strip()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_GEN_MODEL", "llama3.1")
        answers: List[str] = []
        cumulative = []
        for step in plan.ordered_steps:
            cumulative.append(step.text)
            
            vector_terms = Orchestrator._vector_terms(
                    query=step.text,
                    faiss_dir="faiss_store"
                )
            if self.debug:
                    answers.append(f"[{step.id}] vector_terms={vector_terms}")
            if vector_terms:
                    step.text += "\n\nAdditional Context Terms: " + ", ".join(vector_terms)
            ans = f"[{step.id}] processed -> {step.text}"
            if self.debug:
                ans += f" | context_so_far={len(cumulative)}"
            answers.append(ans)
            return answers
    

    def _vector_terms(query: str, top_k: int = 8, faiss_dir: str = "faiss_store") -> List[str]:
        if FaissVectorStoreCosine is None:
            return []
        try:
            store = FaissVectorStoreCosine(persist_dir=faiss_dir)
            try:
                print("dir exists:", os.path.isdir(store.persist_dir))
                print("files:", os.listdir(store.persist_dir))
                store.load()
                print("loaded")
                result = store.query("Identify donors in Garciafurt.", 100)
                print("hits:", len(result))
            except Exception as e:
                import traceback; traceback.print_exc()
            hits = store.query(query, 100) or []
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
