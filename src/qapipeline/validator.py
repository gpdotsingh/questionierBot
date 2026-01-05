from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    CompilerOutput,
    OrchestratorOutput,
    Plan,
    ValidatorInput,
    ValidatorOutput,
)


class Validator:
    """
    Cross-checks splitter, orchestrator, and compiler outputs.
    Computes a confidence score and assembles a JSON-ready response with
    answer, reasoning, and citations (queries + outputs).
    """

    def __init__(self, min_confidence: float = 0.45) -> None:
        self.min_confidence = min_confidence

    def _keyword_set(self, text: str) -> set:
        return {m.lower() for m in re.findall(r"[A-Za-z0-9_]{3,}", text or "")}

    def _sort_key(self, key: str) -> int:
        if isinstance(key, str) and key.startswith("Q") and key[1:].isdigit():
            return int(key[1:])
        return 0

    def _flatten_steps(self, plan: Optional[Plan]) -> List[str]:
        tree = None
        if plan is None:
            return []
        if isinstance(plan, Plan):
            tree = plan.ordered_steps
        elif isinstance(plan, dict):
            tree = plan
        if not isinstance(tree, dict):
            return []

        steps: List[str] = []

        def walk(qid: str, node: Dict[str, Any]) -> None:
            text = (node or {}).get("text") or ""
            if text:
                steps.append(f"{qid}: {text}")
            for child in (node or {}).get("children", []):
                if isinstance(child, dict):
                    for child_id, child_node in child.items():
                        walk(str(child_id), child_node)

        for k, v in sorted(tree.items(), key=lambda kv: self._sort_key(kv[0])):
            walk(str(k), v)
        return steps

    def _flatten_queries(
        self, orchestrator_output: Optional[OrchestratorOutput]
    ) -> Tuple[List[str], List[Any]]:
        if orchestrator_output is None:
            return [], []
        tree = orchestrator_output.queries or {}
        results = orchestrator_output.query_result or []
        if not isinstance(tree, dict):
            return [], results

        queries: List[str] = []

        def walk(qid: str, node: Dict[str, Any]) -> None:
            text = (node or {}).get("text") or ""
            if text:
                queries.append(text.strip())
            for child in (node or {}).get("children", []):
                if isinstance(child, dict):
                    for child_id, child_node in child.items():
                        walk(str(child_id), child_node)

        for k, v in sorted(tree.items(), key=lambda kv: self._sort_key(kv[0])):
            walk(str(k), v)
        return queries, results

    def _calc_relevance(
        self, question: str, answer: str, steps: List[str]
    ) -> float:
        q_tokens = self._keyword_set(question)
        step_tokens: set = set()
        for s in steps:
            step_tokens |= self._keyword_set(s)
        ref_tokens = q_tokens | step_tokens
        ans_tokens = self._keyword_set(answer)
        if not ref_tokens or not ans_tokens:
            return 0.0
        overlap = len(ans_tokens & ref_tokens)
        return min(1.0, overlap / max(1, len(ref_tokens)))

    def _calc_evidence_score(self, results: List[Any]) -> float:
        if not results:
            return 0.0
        total = len(results)
        good = 0
        for r in results:
            rs = str(r).lower()
            if rs.startswith("error"):
                continue
            good += 1
        return min(1.0, good / total)

    def _calc_completeness(self, answer: str) -> float:
        # Encourage concise but non-empty answers; 50+ words scores 1.0
        tokens = self._keyword_set(answer)
        if not tokens:
            return 0.0
        return min(1.0, len(tokens) / 50.0)

    def _build_response_json(
        self,
        compiled: CompilerOutput,
        steps: List[str],
        queries: List[str],
        results: List[Any],
        score: float,
        valid: bool,
    ) -> Dict[str, Any]:
        citations = []
        for idx, query in enumerate(queries):
            citations.append(
                {
                    "id": idx + 1,
                    "query": query,
                    "output": results[idx] if idx < len(results) else None,
                }
            )
        return {
            "answer": compiled.final_answer,
            "reasoning": steps,
            "citations": citations,
            "confidence": round(score, 3),
            "valid": valid,
        }

    def validate(self, data: ValidatorInput) -> ValidatorOutput:
        compiled = data.compiler_output or CompilerOutput(
            final_answer=data.compiled_answer
        )
        steps = self._flatten_steps(data.plan)
        queries, results = self._flatten_queries(data.orchestrator_output)

        relevance = self._calc_relevance(
            data.original_question, compiled.final_answer, steps
        )
        evidence = self._calc_evidence_score(results)
        completeness = self._calc_completeness(compiled.final_answer)

        score = round(
            (0.45 * relevance) + (0.35 * evidence) + (0.20 * completeness), 3
        )
        valid = score >= self.min_confidence

        response_json = self._build_response_json(
            compiled, steps, queries, results, score, valid
        )
        notes = (
            f"relevance={relevance:.2f}, evidence={evidence:.2f}, "
            f"completeness={completeness:.2f}; "
            f"valid={'yes' if valid else 'no'}"
        )
        return ValidatorOutput(
            score=score,
            notes=notes,
            valid=valid,
            response_json=response_json,
        )
