from __future__ import annotations
from typing import Any, Dict, List, Optional
from .models import Plan , OrchestratorOutput
from .settings import get_provider_runtime, ensure_env_loaded
from .llm_common import LLMRouterBase, LLMJsonMixin
import re
import json
from pymongo import MongoClient
import os
from decimal import Decimal
from datetime import date, datetime, time
# ---------- Regex helpers ----------
JOINERS = re.compile(r"\b(?:and|also|plus|as well as)\b", re.I)
SEQUENCERS = re.compile(r"\b(?:then|next|after|based on|using|with|from)\b", re.I)
SENT_SPLIT = re.compile(r"[.;?!]|\bthen\b", re.I)
try:
    from src.ingestdata.faiss_store import FaissVectorStoreCosine
except Exception:
    FaissVectorStoreCosine = None

def _norm(s: str) -> str:
    return LLMJsonMixin.norm(s)

class _LLMRouter(LLMRouterBase):
    def __init__(self) -> None:
        super().__init__(runtime_name="ORCHESTRATOR")

    @staticmethod
    def _prompt_header() -> str:
        return (
            "You are a query planner for MongoDB using aggregation pipelines. Use Metadata to ground fields.\n"
            "Output ONLY raw JSON (no backticks, no prose) shaped like:\n"
            "{\n"
            "  \"Q1\": {\"text\": [ {\"$match\": {...}}, {\"$group\": {...}} ], \"children\": [ {\"Q2\": {\"text\": [ {\"$match\": {...}} ], \"children\": []}} ]},\n"
            "  \"Q3\": {\"text\": [ {\"$match\": {...}}, {\"$sort\": {...}} ], \"children\": []}\n"
            "}\n"
            "Rules:\n"
            "- Target collection: donors \n"
            "- Each node's \"text\" must be a Mongo pipeline array; may also accept {\"pipeline\": [...]}.\n"
            "- Keep steps minimal and nest dependents as children.\n"
            "- Resolve natural language via Metadata.synonyms when mapping to fields.\n"
            "- No commentary, no code fences. Keys must be Q1..Qn only."
            "- Example { 'FirstName': { '$regex': 'an', '$options': 'i' } }"
        )

    @staticmethod
    def _prompt_body(user_query: str, meta_text: str, hint_text: str = "", memory_text: str = "") -> str:
        block_meta = f"Metadata:\n{meta_text}\n" if meta_text else "Metadata:\n(none)\n"
        block_hint = f"\nVectorHints:\n{hint_text}\n" if hint_text else ""
        block_mem = f"\nConversationMemory:\n{memory_text}\n" if memory_text else ""
        return (
            f"{block_meta}"
            f"{block_mem}"
            f"UserQuery:\n\"{_norm(user_query)}\"\n"
            f"{block_hint}\n"
            "Output JSON now:"
        )



class Orchestrator(LLMJsonMixin):
    """
    Dummy orchestrator: for each step produces a synthetic answer string.
    Input: Plan
    Output: List[str] (answers) passed to compiler.
    """
    def __init__(
            self,
            debug: bool = False,
            try_llm: bool = True,
            faiss_dir: str = "faiss_store",
            question: str = "",
            metadata: Dict[str, Any]={},
            **kwargs,            
        ):
        ensure_env_loaded()
        self.debug = debug
        runtime = get_provider_runtime("orchestrator")
        self.provider = runtime.provider
        self._openai = runtime.openai_client
        self.openai_model = runtime.openai_model
        self._ollama = runtime.ollama_client
        self.ollama_model = runtime.ollama_model

        self.try_llm = try_llm
        self.faiss_dir = faiss_dir
        self.router = _LLMRouter() if try_llm else None

    def run(self, plan: Plan, memory_text: str = "") -> OrchestratorOutput:
        provider = self.provider
        self.provider = provider
        answers: List[str] = []
        self.question = json.dumps(plan.ordered_steps, indent=2)
        self.metadata = plan.metadata
        print(self.question )  # plain JSON without dict_values
        max_attempts = 3
        generated_queries: Optional[Dict[str, Any]] = None
        results_list: List[str] = []
        for attempt in range(1, max_attempts + 1):
            if generated_queries is None:
                generated_queries = self._attempt_llm(self.question, memory_text=memory_text)
            results_list = self._execute_generated_queries(generated_queries)
            # Check for any errors
            has_error = any(isinstance(r, str) and r.startswith("ERROR:") for r in results_list)
            if not has_error:
                break
            error_texts = "\n".join(
                r for r in results_list if isinstance(r, str) and r.startswith("ERROR:")
            )
            # Regenerate queries for next attempt
            generated_queries = self._attempt_llm(
                f"{self.question}\nPrevious errors:\n{error_texts}\nFix previous Mongo pipeline errors (stage order, field names, types).",
                memory_text=memory_text
            )

        return  OrchestratorOutput(
            original_question=plan.original_question,
            metadata=plan.metadata,
            ordered_steps=plan.ordered_steps,
            answers=answers,
            queries=generated_queries,
            query_result=results_list
        )
    
    def _attempt_llm(self, user_query: str, memory_text: str = "") -> Optional[Dict[str, Any]]:
        if not (self.try_llm and self.router and self.router.provider):
            return None
        prompt = _LLMRouter._prompt_header() + "\n" + _LLMRouter._prompt_body(user_query, self.metadata, memory_text=memory_text)
        return self.router.ask_json(prompt)

    def _mongo_connect(self):
        uri = os.getenv("MONGO_URI")
        dbname = os.getenv("MONGO_DB", "chatbot")
        collname = os.getenv("MONGO_COLLECTION", "donors")
        client = MongoClient(uri)
        return client, client[dbname][collname]

    @staticmethod
    def _coerce_pipeline(val: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(val, list) and val and isinstance(val[0], dict):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed
            except Exception:
                return None
        return None
    
    def _extract_pipeline_list(self, tree: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        pipelines: List[List[Dict[str, Any]]] = []
        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            p = self._coerce_pipeline(node.get("text")) or self._coerce_pipeline(node.get("pipeline"))
            if p:
                pipelines.append(p)
            for child in node.get("children") or []:
                if isinstance(child, dict):
                    for _, cv in child.items():
                        walk(cv)
        for _, v in sorted(
            (tree or {}).items(),
            key=lambda kv: int(kv[0][1:]) if isinstance(kv[0], str) and kv[0].startswith("Q") and kv[0][1:].isdigit() else 0
        ):
            walk(v)
        return pipelines

    @staticmethod
    def _json_default(o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (datetime, date, time)):
            return o.isoformat()
        return str(o)

    def _execute_generated_queries(self, generated_queries: Optional[Dict[str, Any]]) -> List[str]:
        """
        Execute each MongoDB aggregation pipeline (in order) and return stringified results.
        If generated_queries is None or empty, returns [].
        """
        print("Generated Queries:", generated_queries)      
        if not isinstance(generated_queries, dict) or not generated_queries:
            return []
        pipelines = self._extract_pipeline_list(generated_queries)
        if not pipelines:
            return []

        results: List[str] = []
        client = None
        try:
            client, coll = self._mongo_connect()
            for pipe in pipelines:
                try:
                    cur = coll.aggregate(pipe)
                    rows = list(cur)
                    results.append(json.dumps(rows, default=self._json_default))
                except Exception as e:
                    results.append(f"ERROR: {e}\nPIPELINE: {json.dumps(pipe)[:1000]}")
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass
        return results
    
