from __future__ import annotations
from typing import Any, Dict, Optional
import json

from qapipeline.llm_common import LLMRouterBase
from splitter import _LLMRouter
from .models import OrchestratorOutput, CompilerOutput


class _LLMRouter(LLMRouterBase):
    def __init__(self) -> None:
        super().__init__(runtime_name="ORCHESTRATOR")

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
        if isinstance(vals, list):
            out = []
            for i, item in enumerate(vals, 1):
                s = item if isinstance(item, str) else json.dumps(item) if isinstance(item, (dict, list)) else str(item)
                out.append(f"- Result {i}: {s[:800]}")
            return "\n".join(out)
        return str(vals)

    # ---- Prompt builder ----
    def _build_prompt(self, output: OrchestratorOutput) -> str:
        oq = (output.original_question or "").strip()
        meta_str = self._meta_to_text(output.metadata or {})
        results_str = self._results_to_text(output.query_result or {})

        return (
            "You are a report compiler. Write a clear, concise, human-readable answer to the OriginalQuestion.\n"
            "Use Metadata to correctly name entities/fields and QueryResults as the factual basis.\n"
            "Requirements:\n"
            "- Be accurate and brief (3–8 sentences or short bullet points).\n"
            "- Include key numbers, top entries (limit 3–5), and relevant filters (city/state/date ranges).\n"
            "- If results are empty or contain errors, state that and suggest a correction.\n"
            "- Plain text only. No code fences, no SQL, no JSON.\n\n"
            f"OriginalQuestion:\n{oq}\n\n"
            f"Metadata:\n{meta_str}\n\n"
            f"QueryResults:\n{results_str}\n\n"
            "Now write the final answer:"
        )

    # ---- LLM attempt ----
    def _attempt_llm(self, output: OrchestratorOutput) -> Optional[str]:
        if not (self.try_llm and self.router and self.router.provider):
            return None
        prompt = self._build_prompt(output)
        raw = self.router.ask_json(prompt)  # If router enforces JSON, adjust to ask text instead
        # If ask_json returns dict, try extracting a 'final' or join keys; else return None.
        if isinstance(raw, dict):
            # Prefer a 'final' key; otherwise join string values
            final = raw.get("final") or "\n".join(str(v) for v in raw.values() if isinstance(v, str))
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