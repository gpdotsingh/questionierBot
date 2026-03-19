from __future__ import annotations
from typing import Any, Dict, Optional
import json

from .models import OrchestratorOutput, CompilerOutput
from .llm_common import LLMRouterBase, LLMJsonMixin


def _norm(s: str) -> str:
    return LLMJsonMixin.norm(s)
    
class _LLMRouter(LLMRouterBase):
    def __init__(self) -> None:
        super().__init__(runtime_name="COMPILER")

        # ---- Prompt builder ----
    @staticmethod
    def _prompt_header(output: OrchestratorOutput) -> str:
        oq = (output.original_question or "").strip()
        
        return (
            "You are a report compiler. Write a clear, concise,  to the OriginalQuestion.\n"
            "Use Metadata to correctly name entities/fields and QueryResults as the factual basis.\n"
            "Requirements:\n"
            "- Be accurate and brief (3–8 sentences or short bullet points).\n"
            "- Include key numbers, top entries (limit 3–5), and relevant filters (city/state/date ranges).\n"
            "- If results are empty or contain errors, state that and suggest a correction.\n"
            "- Plain text only. No code fences, no SQL, no JSON.\n\n"
            f"OriginalQuestion:\n{oq}\n\n"
            "Now write the final answer.\n"
            'Output as JSON with a single key "final" containing your answer string, e.g. {"final": "Your answer here"}:\n'
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

class LLMCompiler:
    def __init__(self, try_llm: bool = True, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.try_llm = try_llm
        self.router = _LLMRouter() if try_llm else None
        self.metadata = metadata or {}

    # ---- Formatting helpers ----
    def _meta_to_text(self, md: Dict[str, Any]) -> str:
        if not md:
            return "(none)"
        lines = []
        ds = md.get("dataset") or {}
        fields = md.get("fields") or {}
        syn = md.get("synonyms") or {}
        if ds:
            lines.append("Dataset:")
            for k, v in ds.items():
                lines.append(f"- {k}: {v}")
        if fields:
            lines.append("Fields:")
            for k, v in fields.items():
                lines.append(f"- {k}: {v}")
        if syn:
            lines.append("Synonyms:")
            for k, v in syn.items():
                vv = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
                lines.append(f"- {k}: {vv}")
        return "\n".join(lines)

    def _results_to_text(self, qr: Dict[str, Any]) -> str:
        if not qr:
            return "(none)"

        vals = None
        if isinstance(qr, dict):
            vals = qr.get("results") or qr.get("data") or qr
        else:
            vals = qr

        if not isinstance(vals, list):
            return str(vals)

        out = []
        for i, item in enumerate(vals, 1):
            # Each item in query_result is a JSON array string (all records for one entity).
            # Parse it so every individual record is shown to the LLM, not just the first 800 chars.
            if isinstance(item, str) and item.strip().startswith("["):
                try:
                    records = json.loads(item)
                    if isinstance(records, list):
                        out.append(f"QueryResult {i} ({len(records)} records):")
                        for j, rec in enumerate(records, 1):
                            rec_s = json.dumps(rec) if isinstance(rec, dict) else str(rec)
                            out.append(f"  [{j}] {rec_s[:600]}")
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
            # Fallback: plain string or dict — show up to 2000 chars
            s = item if isinstance(item, str) else json.dumps(item) if isinstance(item, (dict, list)) else str(item)
            out.append(f"- Result {i}: {s[:2000]}")
        return "\n".join(out)

    
    # ---- LLM attempt ----
    def _attempt_llm(self, output: OrchestratorOutput) -> Optional[str]:
        if not (self.try_llm and self.router and self.router.provider):
            return None
        
        meta = self._meta_to_text(output.metadata or {})
        results = self._results_to_text(output.query_result or {})
        body = _LLMRouter._prompt_body(output.original_question or "",meta, results)
        header = _LLMRouter._prompt_header(output=output)
        prompt = header + body
        raw = self.router.ask_json(prompt)
        print(f"[DEBUG compiler] raw from LLM: {raw}")

        # If ask_json returns dict, try extracting a known answer key; else join all values.
        if isinstance(raw, dict):
            # Prefer common answer keys the LLM might use
            final = (
                raw.get("final")
                or raw.get("answer")
                or raw.get("response")
                or raw.get("final_answer")
                or raw.get("summary")
            )
            # If none of the known keys matched, join ALL values (not just strings)
            if not final:
                final = "\n".join(str(v) for v in raw.values() if v)
            return final or None
        # If router returns text, pass through
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    # ---- Fallback synthesis ----
    def _fallback(self, output: OrchestratorOutput) -> str:
        joined = "\n".join(output.answers or [])
        return f"Question: {output.original_question}\nSynthesized Answer:\n{joined}"

    # ---- Public compile method ----
    def compile(self, data: OrchestratorOutput) -> CompilerOutput:
        llm_text = self._attempt_llm(data)
        final = llm_text if llm_text else self._fallback(data)
        return CompilerOutput(final_answer=final, details={"answer_count": len(data.answers)})