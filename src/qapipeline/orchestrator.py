from __future__ import annotations
from typing import Any, Dict, List, Optional
from .models import Plan , OrchestratorOutput
from .settings import get_provider_runtime, ensure_env_loaded
from .llm_common import LLMRouterBase, LLMJsonMixin
import re
import json
import psycopg2
import psycopg2.extras
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
            "You are a query planner for a SQL database. Use the provided Metadata to ground all fields.\n"
            "Output ONLY raw JSON (no backticks, no prose) in this exact shape:\n"
            "{\n"
            "  \"Q1\": {\"text\": \"...\", \"children\": [ {\"Q2\": {\"text\": \"...\", \"children\": []}}, {\"Q3\": {\"text\": \"...\", \"children\": []}} ]},\n"
            "  \"Q4\": {\"text\": \"...\", \"children\": []}\n"
            "}\n"
            "Rules:\n"
            "- Target table MUST be \"donors\".\n"
            "- Column names MUST be the snake_case of Metadata.fields KEYS (NOT their display values).\n"
            "  Examples:\n"
            "    donor_id -> column \"donor_id\"\n"
            "    first_name -> column \"first_name\"\n"
            "    total_amount_donated -> column \"total_amount_donated\"\n"
              "Date arithmetic and EXTRACT (PostgreSQL):\n"
            "  - Days since date: (current_date - \"last_donation_date\"::date) AS recency_days (integer).\n"
            "  - EXTRACT is ONLY valid on timestamp or interval; NEVER call EXTRACT on integers.\n"
            "  - If EXTRACT(day) is needed, use an interval: EXTRACT(day FROM age(current_date, \"last_donation_date\"::date)).\n"
            "  - Buckets: date_trunc('month', \"last_donation_date\"::timestamp) AS month; GROUP BY month.\n"
            "  - Add/subtract durations with intervals: current_date - interval '30 days'.\n"
            "  - Cast explicitly when needed: \"last_donation_date\"::date or ::timestamp.\n"
            
            "Text functions (TRIM/LENGTH/LOWER/UPPER):\n"
            "  - Apply only to text columns. Do NOT call TRIM/LENGTH on integer/numeric.\n"
            "  - Cast non-text when needed: TRIM(CAST(\"zip_code\" AS text)), LENGTH(CAST(\"zip_code\" AS text)).\n"
            "  - Example: (LENGTH(TRIM(\"city\")) > 0) AND (\"zip_code\" IS NULL OR LENGTH(TRIM(CAST(\"zip_code\" AS text))) = 0).\n"
            "- Use proper PostgreSQL date arithmetic (intervals), not date + numeric."
            "Boolean fields:\n"
            "  - Never use empty string '' or numeric with COALESCE on boolean columns.\n"
            "  - Use COALESCE(\"event_participation\", false) for null-safe boolean.\n"
            "  - If a numeric flag is needed: CASE WHEN \"event_participation\" THEN 1 ELSE 0 END AS event_participation_int.\n"
            "  - If a text label is needed: CASE WHEN \"event_participation\" THEN 'yes' ELSE 'no' END AS event_participation_label.\n"
            "- Resolve natural language using Metadata.synonyms mappings.\n"
            "- Keep steps minimal and executable. If a step depends on a previous result, nest it as a child.\n"
            "- Include validation steps when an entity is referenced (e.g., check City/State exists).\n"
            "- Prefer a single root with children unless independent roots are clearly separate.\n"
            "- Return sequesnce of queries if required \n"
            "- If multiple queries can be combined in one query then combine one all in single query \n"
            "- Do not include commentary or code fences. Keys must be Q1..Qn only.\n"
            "\n"
            "After this header you will receive:\n"
            
            "- Metadata (YAML)\n"
            "- UserQuestion\n"
            "Then emit the JSON now."
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

    # ask_json provided by LLMRouterBase

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

    def run(self, plan: Plan) -> OrchestratorOutput:
        provider = self.provider
        self.provider = provider
        answers: List[str] = []
        self.question = json.dumps(plan.ordered_steps, indent=2)
        self.metadata = plan.metadata
        print(self.question )  # plain JSON without dict_values
        generated_queries = self._attempt_llm(self.question)

        print(generated_queries)
        cumulative = []
        results = self._execute_generated_queries(generated_queries)
        return  OrchestratorOutput(
            original_question=plan.original_question,
            metadata=plan.metadata,
            ordered_steps=plan.ordered_steps,
            answers=answers,
            queries=generated_queries,
            query_result=results
        )
    
    def _attempt_llm(self, user_query: str) -> Optional[Dict[str, Any]]:
        if not (self.try_llm and self.router and self.router.provider):
            return None
        prompt = _LLMRouter._prompt_header() + "\n" + _LLMRouter._prompt_body(user_query, self.metadata)
        raw = self.router.ask_json(prompt)
        return raw

    def _pg_connect(self):
        """
        Create a Postgres connection using .env values.
        """
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "127.0.0.1"),
            port=os.getenv("PG_PORT", "5432"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", ""),
            dbname=os.getenv("PG_DATABASE", "postgres"),
            sslmode=os.getenv("PG_SSLMODE", "disable"),
        )
        return conn

    def _extract_sql_list(self, tree: Dict[str, Any]) -> List[str]:
        """
        Traverse the generated_queries tree and return a list of SQL texts in sequence (DFS).
        Accepts format:
          { "Q1": {"text": "...", "children": [ {"Q2": {...}}, {"Q3": {...}} ] }, "Q4": {...} }
        """
        sqls: List[str] = []
        def walk(node_val: Dict[str, Any]):
            txt = (node_val or {}).get("text", "")
            if isinstance(txt, str) and txt.strip():
                sqls.append(txt.strip())
            for child in (node_val or {}).get("children", []):
                for _, cv in child.items():
                    walk(cv)
        # sort roots by Q index
        for k, v in sorted(tree.items(), key=lambda kv: int(kv[0][1:]) if kv[0].startswith("Q") and kv[0][1:].isdigit() else 0):
            walk(v)
        return sqls 

    @staticmethod
    def _json_default(o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (datetime, date, time)):
            return o.isoformat()
        return str(o)

    def _execute_generated_queries(self, generated_queries: Optional[Dict[str, Any]]) -> List[str]:
        """
        Execute each SQL query (in order) against Postgres and return stringified results.
        If generated_queries is None or empty, returns [].
        """
        if not isinstance(generated_queries, dict) or not generated_queries:
            return []
        sqls = self._extract_sql_list(generated_queries)
        if not sqls:
            return []

        results: List[str] = []
        conn = None
        try:
            conn = self._pg_connect()
            with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                for sql in sqls:
                    try:
                        cur.execute(sql)
                        # If SELECT, fetch rows; else return rowcount/command status
                        if cur.description:
                            rows = cur.fetchall()
                            results.append(json.dumps(rows, default=self._json_default))
                        else:
                            results.append(f"OK rows={cur.rowcount}")
                    except Exception as e:
                        results.append(f"ERROR: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        return results
    # ask_json provided by LLMJsonMixin
