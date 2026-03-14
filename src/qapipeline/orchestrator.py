from __future__ import annotations
from typing import Any, Dict, List, Optional
from .models import Plan , OrchestratorOutput
from .settings import get_provider_runtime, ensure_env_loaded
from .llm_common import LLMRouterBase, LLMJsonMixin
import re
import json
import requests
import os
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
    def _prompt_header(entity_list: str = "") -> str:
        return (
            "You are a query planner for QuickBooks Online IDS Query API.\n"
            "Output ONLY raw JSON (no backticks, no prose) shaped like:\n"
            "{\n"
            "  \"Q1\": {\"text\": \"select * from Invoice where Balance > '0'\", \"children\": [\n"
            "    {\"Q2\": {\"text\": \"select * from Customer\", \"children\": []}}\n"
            "  ]},\n"
            "  \"Q3\": {\"text\": \"select * from Bill where TotalAmt > '500'\", \"children\": []}\n"
            "}\n"
            "Rules:\n"
            "- Each node's \"text\" must be a single QuickBooks IDS query string.\n"
            "- IDS query syntax: SELECT * FROM <Entity> [WHERE <conditions>] [ORDERBY <field> [ASC|DESC]] [STARTPOSITION n] [MAXRESULTS n]\n"
            "- WHERE clause operators: =, <, >, <=, >=, LIKE, IN\n"
            "- String values must be single-quoted: WHERE DisplayName = 'John Smith'\n"
            "- Date values use format: WHERE TxnDate > '2025-01-01'\n"
            "- Numeric values are unquoted: WHERE TotalAmt > 1000\n"
            "- Boolean values: WHERE Active = true\n"
            "- LIKE uses '%' wildcard: WHERE DisplayName LIKE '%smith%'\n"
            "- IN uses parentheses: WHERE Id IN ('1', '2', '3')\n"
            "- No $match, $group, $sort or MongoDB syntax. IDS queries only.\n"
            "- IDS does NOT support JOIN, GROUP BY, SUM, COUNT, AVG or subqueries.\n"
            "- To aggregate, fetch raw data and let the downstream compiler handle it.\n"
            "- If asking for 'top N' or 'highest', use ORDERBY <field> DESC MAXRESULTS N.\n"
            f"- Available entities: {entity_list}\n"
            "- Keep steps minimal and nest dependents as children.\n"
            "- Resolve natural language via Metadata.synonyms when mapping to fields.\n"
            "- No commentary, no code fences. Keys must be Q1..Qn only.\n"
            "- Default MAXRESULTS to 100 unless user asks for specific count.\n"
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
    Orchestrator: generates QuickBooks IDS queries via LLM and executes them
    against the Spring Boot QuickBooks API.
    Input: Plan
    Output: OrchestratorOutput with query results passed to compiler.
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
                f"{self.question}\nPrevious errors:\n{error_texts}\nFix previous QuickBooks IDS query errors (entity names, field names, query syntax, WHERE clause operators).",
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

    def _get_entity_list(self) -> str:
        """Extract available entity names from metadata for prompt context."""
        if isinstance(self.metadata, dict):
            entities = self.metadata.get("entities")
            if isinstance(entities, dict):
                return ", ".join(sorted(entities.keys()))
            fields = self.metadata.get("fields")
            if isinstance(fields, dict):
                entity_set = set()
                for v in fields.values():
                    if isinstance(v, str) and "." in v:
                        entity_set.add(v.split(".")[0])
                if entity_set:
                    return ", ".join(sorted(entity_set))
        return (
            "Account, Bill, BillPayment, CreditMemo, Customer, Deposit, "
            "Employee, Estimate, Invoice, Item, JournalEntry, Payment, "
            "PaymentMethod, Purchase, PurchaseOrder, RefundReceipt, "
            "SalesReceipt, TaxAgency, TaxCode, TaxRate, Term, TimeActivity, Vendor"
        )

    def _attempt_llm(self, user_query: str, memory_text: str = "") -> Optional[Dict[str, Any]]:
        if not (self.try_llm and self.router and self.router.provider):
            return None
        entity_list = self._get_entity_list()
        meta_text = self.metadata if isinstance(self.metadata, str) else json.dumps(self.metadata, default=str)
        prompt = _LLMRouter._prompt_header(entity_list=entity_list) + "\n" + _LLMRouter._prompt_body(user_query, meta_text, memory_text=memory_text)
        return self.router.ask_json(prompt)

    def _extract_query_list(self, tree: Dict[str, Any]) -> List[str]:
        """Walk the Q-tree and extract each IDS query string."""
        queries: List[str] = []
        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            text = node.get("text")
            if isinstance(text, str) and text.strip():
                queries.append(text.strip())
            for child in node.get("children") or []:
                if isinstance(child, dict):
                    for _, cv in child.items():
                        walk(cv)
        for _, v in sorted(
            (tree or {}).items(),
            key=lambda kv: int(kv[0][1:]) if isinstance(kv[0], str) and kv[0].startswith("Q") and kv[0][1:].isdigit() else 0
        ):
            walk(v)
        return queries

    @staticmethod
    def _extract_entity_data(raw_json: str) -> str:
        """
        Parse the QuickBooks API response and extract the entity array.
        API returns: {"data": {"QueryResponse": {"<Entity>": [...], ...}, ...}}
        Returns the entity array as a JSON string.
        """
        try:
            parsed = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return raw_json

        data = parsed
        if isinstance(parsed, dict) and "data" in parsed:
            data = parsed["data"]
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    return raw_json

        query_response = data.get("QueryResponse") if isinstance(data, dict) else None
        if not isinstance(query_response, dict):
            return json.dumps(data)

        skip_keys = {"maxResults", "startPosition", "totalCount"}
        for key, value in query_response.items():
            if key not in skip_keys and isinstance(value, list):
                return json.dumps(value)

        return json.dumps(query_response)

    def _execute_generated_queries(self, generated_queries: Optional[Dict[str, Any]]) -> List[str]:
        """
        Execute each QuickBooks IDS query via HTTP POST to the Spring Boot API
        and return stringified results.
        """
        print("Generated Queries:", generated_queries)
        if not isinstance(generated_queries, dict) or not generated_queries:
            return []

        queries = self._extract_query_list(generated_queries)
        if not queries:
            return []

        base_url = os.getenv("QB_API_URL", "http://localhost:8080")
        jwt_token = os.getenv("QB_JWT_TOKEN", "")
        if not jwt_token:
            return ["ERROR: QB_JWT_TOKEN environment variable is not set. Please authenticate via OAuth flow first."]

        url = f"{base_url.rstrip('/')}/api/query"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        results: List[str] = []
        session = requests.Session()
        session.headers.update(headers)

        try:
            for query_text in queries:
                try:
                    resp = session.post(url, json={"query": query_text}, timeout=30)
                    resp.raise_for_status()
                    entity_data = self._extract_entity_data(resp.text)
                    results.append(entity_data)
                except requests.exceptions.HTTPError as e:
                    error_body = ""
                    try:
                        error_body = e.response.text[:500]
                    except Exception:
                        pass
                    results.append(f"ERROR: HTTP {e.response.status_code} - {error_body}\nQUERY: {query_text[:500]}")
                except requests.exceptions.ConnectionError as e:
                    results.append(f"ERROR: Connection failed to {url} - {e}\nQUERY: {query_text[:500]}")
                except requests.exceptions.Timeout:
                    results.append(f"ERROR: Request timed out\nQUERY: {query_text[:500]}")
                except Exception as e:
                    results.append(f"ERROR: {e}\nQUERY: {query_text[:500]}")
        finally:
            session.close()

        return results
