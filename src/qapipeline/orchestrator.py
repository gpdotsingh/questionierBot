from __future__ import annotations
from typing import Any, Dict, List, Optional
from .models import Plan , OrchestratorOutput
from .settings import ensure_env_loaded
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

# Strips anything after the entity name so IDS text is always "select * from <Entity>"
_ENTITY_ONLY = re.compile(r'^\s*select\s+\*\s+from\s+(\w+)', re.IGNORECASE)

def _normalize_ids_query(text: str) -> str:
    """Ensure the IDS query text is strictly 'select * from <Entity>'."""
    m = _ENTITY_ONLY.match(text.strip())
    return f"select * from {m.group(1)}" if m else text.strip()

class _LLMRouter(LLMRouterBase):
    def __init__(self) -> None:
        super().__init__(runtime_name="ORCHESTRATOR")

    @staticmethod
    def _prompt_header(entity_list: str = "") -> str:
        return (
            "You are a query planner for QuickBooks Online IDS Query API.\n"
            "Output ONLY raw JSON (no backticks, no prose) shaped like:\n"
            "{\n"
            "  \"Q1\": {\"text\": \"select * from Invoice\", \"filter\": \"balance > 0 AND txn_status = 'Open'\", \"children\": [\n"
            "    {\"Q2\": {\"text\": \"select * from Customer\", \"filter\": \"\", \"children\": []}}\n"
            "  ]},\n"
            "  \"Q3\": {\"text\": \"select * from Bill\", \"filter\": \"total_amt > 500\", \"children\": []}\n"
            "}\n"
            "Rules:\n"
            "- Each node has exactly two fields: \"text\" and \"filter\".\n"
            "- CRITICAL: \"text\" must ONLY be: select * from <EntityName>  — nothing else.\n"
            "  Do NOT add WHERE, ORDERBY, STARTPOSITION, MAXRESULTS or any other clause to \"text\".\n"
            "  Correct:   \"text\": \"select * from Invoice\"\n"
            "  WRONG:     \"text\": \"select * from Invoice where Balance > '0'\"\n"
            "  WRONG:     \"text\": \"select * from Invoice ORDERBY TxnDate DESC MAXRESULTS 10\"\n"
            "- All filtering and sorting logic goes into the \"filter\" field, NOT into \"text\".\n"
            f"- Available entities: {entity_list}\n"
            "- Keep steps minimal and nest dependents as children.\n"
            "- No $match, $group, $sort or MongoDB syntax.\n"
            "- IDS does NOT support JOIN, GROUP BY, SUM, COUNT, AVG or subqueries.\n"
            "- To aggregate, fetch raw data and let the downstream compiler handle it.\n"
            "- CRITICAL: Only use field names listed in the entity reference below. "
            "Check valid_values for categorical fields.\n"
            "- CRITICAL: Purchase (expense transactions, has PaymentType) and PurchaseOrder (purchase orders, has POStatus) are DIFFERENT entities.\n"
            "- CRITICAL: Purchase (expense transactions, has PaymentType) and PurchaseOrder "
            "- No commentary, no code fences. Keys must be Q1..Qn only.\n"
            "\n"
            "\"filter\" field — WHERE clause applied on cached results after fetch:\n"
            "  Operators : =  !=  >  <  >=  <=  LIKE  BETWEEN … AND …  IS NULL  IS NOT NULL\n"
            "  Logical   : AND  OR  parentheses for grouping\n"
            "  Field names: camelcase \n"
            "  Strings   : single-quoted   type = 'Service'\n"
            "  Numbers   : unquoted        unit_price > 9.99\n"
            "  Booleans  : unquoted        active = true\n"
            "  Blank     : \"filter\": \"\"   (no filtering — return all fetched records)\n"
            "\n"
            "  Examples:\n"
            "    \"filter\": \"\"                                           → all records\n"
            "    \"filter\": \"type = 'Service'\"                          → equality\n"
            "    \"filter\": \"unit_price > 9.99\"                         → numeric\n"
            "    \"filter\": \"total_amt BETWEEN 100 AND 500\"             → range\n"
            "    \"filter\": \"display_name LIKE '%tech%'\"                → partial match\n"
            "    \"filter\": \"txn_date >= '2025-01-01' AND txn_date <= '2025-12-31'\"  → date range\n"
            "    \"filter\": \"type = 'Service' AND active = true AND unit_price > 5\"  → multi-condition\n"
            "    \"filter\": \"(type = 'Service' OR type = 'Inventory') AND unit_price > 5\"  → grouped\n"
            "    \"filter\": \"vendor_ref_name IS NOT NULL\"               → null check\n"
            "  Resolve natural language via Metadata.synonyms when mapping to fields.\n"
            "  CRITICAL: Fields with valid_values must use only defined values.\n"
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
    Searches the FAISS vector DB for detailed entity/column info from semantics.
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
        self.try_llm = try_llm
        self.faiss_dir = faiss_dir
        self.router = _LLMRouter() if try_llm else None
        self._faiss_store = None

    # ---------- Vector DB search for semantic details ----------
    def _get_faiss_store(self):
        """Lazily load the FAISS store once."""
        if self._faiss_store is not None:
            return self._faiss_store
        if FaissVectorStoreCosine is None:
            return None
        try:
            store = FaissVectorStoreCosine(persist_dir=self.faiss_dir)
            store.load()
            self._faiss_store = store
            return store
        except Exception:
            return None

    def _extract_entities_from_plan(self, ordered_steps: Dict[str, Any]) -> List[str]:
        """Extract entity names from the Q-tree nodes (set by splitter)."""
        entities: List[str] = []
        seen = set()
        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            for ename in (node.get("entities") or []):
                if ename not in seen:
                    seen.add(ename)
                    entities.append(ename)
            for child in (node.get("children") or []):
                if isinstance(child, dict):
                    for _, cv in child.items():
                        walk(cv)
        for _, v in (ordered_steps or {}).items():
            walk(v)
        return entities

    @staticmethod
    def _get_entity_name_from_hit(md: Dict[str, Any]) -> str:
        """Extract entity_name from FAISS hit metadata (may be in _raw after _map_meta)."""
        ename = md.get("entity_name", "")
        if not ename:
            raw = md.get("_raw") or {}
            ename = raw.get("entity_name", "")
        return ename

    def _search_semantics(self, entity_names: List[str], user_query: str = "") -> str:
        """
        Search the FAISS vector DB for detailed entity info (fields, types, valid_values).
        Returns a text reference that the LLM can use to build correct queries.
        """
        store = self._get_faiss_store()
        if store is None:
            return ""

        entity_texts: Dict[str, str] = {}

        # Search by entity name for each entity from the plan
        for ename in entity_names:
            hits = store.query(f"Entity: {ename}", k=3) or []
            for h in hits:
                md = h.get("metadata") or {}
                hit_entity = self._get_entity_name_from_hit(md)
                if hit_entity == ename:
                    # Found exact entity match — use its page_content
                    text = md.get("text", "")
                    if text and ename not in entity_texts:
                        entity_texts[ename] = text
                    break

        # Also search by user query to catch entities not explicitly named
        if user_query:
            hits = store.query(user_query, k=5) or []
            for h in hits:
                md = h.get("metadata") or {}
                hit_entity = self._get_entity_name_from_hit(md)
                if hit_entity and not hit_entity.startswith("_") and hit_entity not in entity_texts:
                    text = md.get("text", "")
                    if text:
                        entity_texts[hit_entity] = text

        if not entity_texts:
            return ""

        lines: List[str] = ["Entity Reference (from semantic database):"]
        for ename, text in entity_texts.items():
            lines.append(f"\n{text}")
        return "\n".join(lines)

    def run(self, plan: Plan, memory_text: str = "") -> OrchestratorOutput:
        answers: List[str] = []
        self.question = json.dumps(plan.ordered_steps, indent=2)
        self.metadata = plan.metadata
        self.ordered_steps = plan.ordered_steps
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

        # Extract entity names from the plan's Q-tree
        plan_entities = self._extract_entities_from_plan(self.ordered_steps) if hasattr(self, 'ordered_steps') and self.ordered_steps else []

        # Search FAISS vector DB for detailed entity info (fields, types, valid_values)
        semantic_ref = self._search_semantics(plan_entities, user_query=user_query)

        # Use semantic reference from vector DB as metadata for the prompt
        if semantic_ref:
            meta_text = semantic_ref
        else:
            # Fallback: use table metadata entities as basic reference
            meta_text = self._build_table_metadata_reference()

        prompt = _LLMRouter._prompt_header(entity_list=entity_list) + "\n" + _LLMRouter._prompt_body(user_query, meta_text, memory_text=memory_text)
        return self.router.ask_json(prompt)

    def _build_table_metadata_reference(self) -> str:
        """Fallback: build entity reference from table metadata when vector DB is unavailable."""
        if not isinstance(self.metadata, dict):
            return ""
        entities = self.metadata.get("entities")
        if not isinstance(entities, dict):
            return json.dumps(self.metadata, default=str)
        lines: List[str] = []
        for name in sorted(entities.keys()):
            info = entities[name]
            if not isinstance(info, dict):
                continue
            desc = info.get("description", "")
            key_cols = info.get("key_columns", [])
            cols_str = ", ".join(key_cols) if key_cols else ""
            lines.append(f"{name}: {desc} | Columns: {cols_str}")
        return "\n".join(lines)

    def _extract_query_list(self, tree: Dict[str, Any]) -> List[Dict[str, str]]:
        """Walk the Q-tree and extract each node as {"text": <IDS query>, "filter": <WHERE clause>}.
        'filter' is the optional cache WHERE clause sent alongside the IDS query to /api/query.
        """
        nodes: List[Dict[str, str]] = []
        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            text = node.get("text")
            if isinstance(text, str) and text.strip():
                nodes.append({
                    "text": _normalize_ids_query(text),   # always "select * from <Entity>"
                    "filter": node.get("filter", "") or "",
                })
            for child in node.get("children") or []:
                if isinstance(child, dict):
                    for _, cv in child.items():
                        walk(cv)
        for _, v in sorted(
            (tree or {}).items(),
            key=lambda kv: int(kv[0][1:]) if isinstance(kv[0], str) and kv[0].startswith("Q") and kv[0][1:].isdigit() else 0
        ):
            walk(v)
        return nodes

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
            for node in queries:
                query_text = node["text"]
                where_filter = node.get("filter", "")
                # Build request body: always send query, add filter only when non-empty
                payload: Dict[str, str] = {"query": query_text}
                if where_filter:
                    payload["filter"] = where_filter
                try:
                    resp = session.post(url, json=payload, timeout=30)
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
