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
            "You are a query planner for QuickBooks Online.\n"
            "Output ONLY raw JSON (no backticks, no prose) shaped like:\n"
            "{\n"
            "  \"Q1\": {\"text\": \"select * from Invoice\", \"filter\": \"balance > 0\", \"children\": []},\n"
            "  \"Q2\": {\"text\": \"select * from Bill\",    \"filter\": \"total_amt > 500\",  \"children\": []}\n"
            "}\n"
            "\n"
            "━━━ SECTION 1 — IDS TRANSACTION QUERIES ━━━\n"
            "Use for fetching individual records (invoices, customers, items, etc.).\n"
            f"Available entities: {entity_list}\n"
            "Rules:\n"
            "- \"text\" MUST be: select * from <EntityName>  — nothing else.\n"
            "  WRONG: select * from Invoice where Balance > 0\n"
            "  WRONG: select * from Invoice ORDERBY TxnDate DESC\n"
            "- All filtering logic goes into \"filter\" (SQL WHERE syntax, see below).\n"
            "- IDS does NOT support JOIN, GROUP BY, SUM, COUNT, AVG or subqueries.\n"
            "- To aggregate, fetch raw data; the downstream compiler handles totals.\n"
            "- CRITICAL: Purchase (expense, has PaymentType) ≠ PurchaseOrder (has POStatus).\n"
            "- Use only field names from the entity reference. Check valid_values.\n"
            "- Invoice records already include customer_ref_name and customer_ref_value —\n"
            "  do NOT add a Customer child query when the parent is Invoice.\n"
            "- For 'due in X days', 'overdue', or 'past due': use select * from Invoice\n"
            "  with filter: balance > 0  and NO date filter in the filter field.\n"
            "  The downstream compiler reads the due_date field and handles date arithmetic.\n"
            "\n"
            "\"filter\" — SQL WHERE clause on cached results (blank = no filter):\n"
            "  Operators: =  !=  >  <  >=  <=  LIKE  BETWEEN … AND …  IS NULL  IS NOT NULL\n"
            "  Logic:     AND  OR  ( )\n"
            "  Examples:\n"
            "    \"filter\": \"total_amt BETWEEN 100 AND 500\"\n"
            "    \"filter\": \"txn_date >= '2025-10-01' AND txn_date <= '2025-12-31'\"\n"
            "    \"filter\": \"type = 'Service' AND active = true\"\n"
            "\n"
            "━━━ SECTION 2 — FINANCIAL REPORTS ━━━\n"
            "Use for summary/aggregate financial data (P&L, Balance Sheet, Cash Flow, Aging, etc.).\n"
            "For reports, set \"text\" to  report:<ReportName>  and \"filter\" to URL query params.\n"
            "\n"
            "Supported report names:\n"
            "  ProfitAndLoss      — Income, expenses, net profit (use for P&L, income statement)\n"
            "  BalanceSheet       — Assets, liabilities, equity snapshot\n"
            "  CashFlow           — Operating/investing/financing cash flows\n"
            "  AgedReceivables    — Outstanding customer invoices by age bucket\n"
            "  AgedPayables       — Outstanding vendor bills by age bucket\n"
            "  TransactionList    — List of all transactions in a date range\n"
            "  CustomerBalance    — Balance owed by each customer\n"
            "  VendorBalance      — Balance owed to each vendor\n"
            "  GeneralLedger      — Full general ledger detail\n"
            "  TrialBalance       — Trial balance of all accounts\n"
            "\n"
            "\"filter\" for reports — space-separated key=value pairs (use _ for spaces in values):\n"
            "  date_macro   : Last_Month  Last_Fiscal_Quarter  Last_Fiscal_Year\n"
            "                 This_Month  This_Fiscal_Year  Last_Fiscal_Year\n"
            "                 This_Week   Last_Week  Today\n"
            "  start_date   : YYYY-MM-DD  (use instead of date_macro for custom ranges)\n"
            "  end_date     : YYYY-MM-DD\n"
            "  accounting_method : Accrual  Cash\n"
            "  Always supply a date filter. Default: date_macro=Last_Fiscal_Quarter\n"
            "\n"
            "Report examples:\n"
            "  P&L last quarter:    {\"text\": \"report:ProfitAndLoss\", \"filter\": \"date_macro=Last_Fiscal_Quarter\", \"children\": []}\n"
            "  P&L last year:       {\"text\": \"report:ProfitAndLoss\", \"filter\": \"date_macro=Last_Fiscal_Year\",    \"children\": []}\n"
            "  Balance Sheet now:   {\"text\": \"report:BalanceSheet\",   \"filter\": \"date_macro=Today\",             \"children\": []}\n"
            "  A/R aging:           {\"text\": \"report:AgedReceivables\",\"filter\": \"date_macro=Today\",             \"children\": []}\n"
            "  Cash basis P&L YTD:  {\"text\": \"report:ProfitAndLoss\", \"filter\": \"date_macro=This_Fiscal_Year accounting_method=Cash\", \"children\": []}\n"
            "  Custom date range:   {\"text\": \"report:ProfitAndLoss\", \"filter\": \"start_date=2025-10-01 end_date=2025-12-31\", \"children\": []}\n"
            "\n"
            "━━━ DECISION RULE ━━━\n"
            "- Revenue / income / total sales / P&L / profit / loss      → report:ProfitAndLoss\n"
            "  For 'this month' or 'this year' revenue: generate TWO queries —\n"
            "    Q1: date_macro=This_Month   (requested period)\n"
            "    Q2: date_macro=Last_Fiscal_Quarter  (context / comparison)\n"
            "  This ensures the compiler can show meaningful data even if the current period\n"
            "  has no transactions yet.\n"
            "- Balance sheet / net worth / assets & liabilities          → report:BalanceSheet\n"
            "- Cash flow statement                                        → report:CashFlow\n"
            "- Outstanding invoices / A/R aging summary                  → report:AgedReceivables\n"
            "- Outstanding bills / A/P aging summary                     → report:AgedPayables\n"
            "- Invoices due soon / overdue / past due / due in X days    → IDS select * from Invoice  (filter: balance > 0, NO date filter)\n"
            "- Individual records (invoices, customers, items, vendors…) → IDS select * from\n"
            "\n"
            "No commentary, no code fences. Keys must be Q1..Qn only.\n"
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
            jwt_token: Optional[str] = None,
            **kwargs,
        ):
        ensure_env_loaded()
        self.debug = debug
        self.try_llm = try_llm
        self.faiss_dir = faiss_dir
        self.router = _LLMRouter() if try_llm else None
        self._faiss_store = None
        # JWT forwarded from the frontend request; takes precedence over QB_JWT_TOKEN env var
        self._jwt_token = jwt_token
        # Set to the new JWT after a successful token refresh (propagated back to the frontend)
        self.refreshed_jwt: Optional[str] = None

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
        """Walk the Q-tree and extract each node.

        Each returned dict has:
          type   : "query" (IDS select) or "report" (QB Reports API)
          text   : IDS query string  — only present for type=query
          report : report name       — only present for type=report
          filter : WHERE clause (query) or space-separated key=value params (report)
        """
        nodes: List[Dict[str, str]] = []

        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            text = node.get("text", "").strip()
            filt = node.get("filter", "") or ""
            if text.lower().startswith("report:"):
                report_name = text.split(":", 1)[1].strip()
                if report_name:
                    nodes.append({"type": "report", "report": report_name, "filter": filt})
            elif text:
                nodes.append({
                    "type": "query",
                    "text": _normalize_ids_query(text),
                    "filter": filt,
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
    def _extract_report_data(raw_json: str) -> str:
        """
        Flatten a QB Reports API response into a readable text summary.
        QB returns: {"Header": {...}, "Columns": {...}, "Rows": {"Row": [...]}}
        Produces human-readable lines the LLM compiler can reason over.
        """
        try:
            parsed = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return raw_json

        # Unwrap {"data": ...} wrapper if present
        if isinstance(parsed, dict) and "data" in parsed:
            inner = parsed["data"]
            if isinstance(inner, str):
                try:
                    parsed = json.loads(inner)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(inner, dict):
                parsed = inner

        header = parsed.get("Header", {})
        report_name = header.get("ReportName", "Report")
        start_period = header.get("StartPeriod", "")
        end_period = header.get("EndPeriod", "")
        basis = header.get("ReportBasis", "")
        currency = header.get("Currency", "")

        lines: List[str] = [
            f"=== {report_name} ===",
            f"Period: {start_period} to {end_period}" if start_period else "",
            f"Basis: {basis}  Currency: {currency}" if basis else "",
            "",
        ]
        lines = [l for l in lines if l is not None]

        def flatten_rows(rows_node: Any, indent: int = 0) -> None:
            if not isinstance(rows_node, dict):
                return
            row_list = rows_node.get("Row", [])
            if not isinstance(row_list, list):
                row_list = [row_list]
            pad = "  " * indent
            for row in row_list:
                if not isinstance(row, dict):
                    continue
                row_type = row.get("type", "")
                # Section header
                hdr = row.get("Header")
                if isinstance(hdr, dict):
                    col_data = hdr.get("ColData", [])
                    label = col_data[0].get("value", "") if col_data else ""
                    if label:
                        lines.append(f"{pad}--- {label} ---")
                # Detail rows
                col_data = row.get("ColData")
                if isinstance(col_data, list) and len(col_data) >= 2:
                    label = col_data[0].get("value", "").strip()
                    value = col_data[1].get("value", "").strip() if len(col_data) > 1 else ""
                    if label and value:
                        lines.append(f"{pad}{label}: {value}")
                # Nested rows
                nested = row.get("Rows")
                if isinstance(nested, dict):
                    flatten_rows(nested, indent + 1)
                # Summary line
                summ = row.get("Summary")
                if isinstance(summ, dict):
                    col_data = summ.get("ColData", [])
                    label = col_data[0].get("value", "").strip() if col_data else ""
                    value = col_data[1].get("value", "").strip() if len(col_data) > 1 else ""
                    if label and value:
                        lines.append(f"{pad}>>> {label}: {value}")
                    lines.append("")

        rows_node = parsed.get("Rows", {})
        flatten_rows(rows_node)
        return "\n".join(lines)

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

        # Resolve JWT: prefer the token forwarded from the frontend request,
        # fall back to the static env-var token (for CLI / dev use).
        auth_header: str = ""
        if self._jwt_token:
            # Already in "Bearer <token>" format from the HTTP Authorization header
            auth_header = self._jwt_token if self._jwt_token.startswith("Bearer ") else f"Bearer {self._jwt_token}"
        else:
            env_token = os.getenv("QB_JWT_TOKEN", "")
            if not env_token:
                return ["ERROR: No JWT token available. Log in via the frontend or set QB_JWT_TOKEN env var."]
            auth_header = f"Bearer {env_token}"

        url = f"{base_url.rstrip('/')}/api/query"
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        results: List[str] = []
        session = requests.Session()
        session.headers.update(headers)

        def _capture_refreshed_jwt(resp: requests.Response) -> None:
            """
            Spring Boot sets X-Refreshed-JWT when it silently refreshed the QB
            access token on our behalf. Capture it so we can propagate it back to
            the frontend, and update the session so remaining queries in this run
            also use the new token.
            """
            new_jwt = resp.headers.get("X-Refreshed-JWT")
            if new_jwt:
                self.refreshed_jwt = new_jwt
                session.headers.update({"Authorization": f"Bearer {new_jwt}"})

        try:
            for node in queries:
                node_type = node.get("type", "query")

                if node_type == "report":
                    # ── QB Reports API: GET /api/reports/{reportName}?key=value&... ──
                    report_name = node["report"]
                    filter_str  = node.get("filter", "") or ""
                    # Parse "key=value key2=value2" — replace _ in values with space
                    params: Dict[str, str] = {}
                    for part in filter_str.split():
                        if "=" in part:
                            k, v = part.split("=", 1)
                            params[k] = v.replace("_", " ")
                    report_url = f"{base_url.rstrip('/')}/api/reports/{report_name}"
                    try:
                        resp = session.get(report_url, params=params, timeout=30)
                        _capture_refreshed_jwt(resp)
                        resp.raise_for_status()
                        results.append(self._extract_report_data(resp.text))
                    except requests.exceptions.HTTPError as e:
                        error_body = ""
                        try:
                            error_body = e.response.text[:500]
                        except Exception:
                            pass
                        results.append(f"ERROR: HTTP {e.response.status_code} - {error_body}\nREPORT: {report_name}")
                    except requests.exceptions.ConnectionError as e:
                        results.append(f"ERROR: Connection failed to {report_url} - {e}\nREPORT: {report_name}")
                    except requests.exceptions.Timeout:
                        results.append(f"ERROR: Request timed out\nREPORT: {report_name}")
                    except Exception as e:
                        results.append(f"ERROR: {e}\nREPORT: {report_name}")

                else:
                    # ── IDS Query API: POST /api/query ──
                    query_text   = node["text"]
                    where_filter = node.get("filter", "")
                    payload: Dict[str, str] = {"query": query_text}
                    if where_filter:
                        payload["filter"] = where_filter
                    try:
                        resp = session.post(url, json=payload, timeout=30)
                        _capture_refreshed_jwt(resp)
                        resp.raise_for_status()
                        results.append(self._extract_entity_data(resp.text))
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
