from __future__ import annotations
import re, os
from typing import List, Optional, Dict
from .models import Step, Plan

# Simple regex helpers (kept same “wiring” idea)
JOINERS = re.compile(r"\b(?:and|also|plus|as well as)\b", re.I)
SEQUENCERS = re.compile(r"\b(?:then|next|after|based on|using|with|from)\b", re.I)
SENT_SPLIT = re.compile(r"[.;?!]|\bthen\b", re.I)
CLEAN_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    return CLEAN_WS.sub(" ", s or "").strip()

class _DummyLLM:
    """Placeholder; never actually calls a model."""
    def __init__(self, enabled: bool):
        self.enabled = enabled
    def plan_hint(self, question: str) -> Optional[List[str]]:
        if not self.enabled:
            return None
        # naive pseudo "LLM" hint: split after commas if any
        parts = [p.strip() for p in question.split(",") if p.strip()]
        return parts if len(parts) > 1 else None

class QuestionSplitter:
    """
    Public API unchanged:
      - __init__(try_llm: bool = True, provider: Optional[str] = None, model: Optional[str] = None)
      - plan(question) -> Plan
    provider/model parameters are accepted (for compatibility) but ignored (env-driven / dummy).
    """
    def __init__(self, try_llm: bool = True, provider: Optional[str] = None, model: Optional[str] = None):
        # We deliberately ignore provider/model; use env to decide enabling dummy LLM
        env_flag = os.getenv("SPLITTER_USE_DUMMY_LLM", "0") in ("1","true","yes")
        self.try_llm = try_llm
        self._llm = _DummyLLM(enabled=env_flag and try_llm)

    def plan(self, question: str) -> Plan:
        q = _norm(question)
        steps: List[Step] = []
        used_llm = False

        # Attempt dummy LLM hint
        llm_parts = self._llm.plan_hint(q) if self.try_llm else None
        if llm_parts:
            used_llm = True
            for i, p in enumerate(llm_parts, 1):
                steps.append(Step(f"S{i}", p))
        else:
            # Rule-based fallback: sentence + joiners
            clauses = [c for c in map(_norm, SENT_SPLIT.split(q)) if c]
            if not clauses:
                clauses = [q]
            sid = 1
            for clause in clauses:
                parts = [p for p in map(_norm, JOINERS.split(clause)) if p]
                if not parts:
                    continue
                sequenced = bool(SEQUENCERS.search(clause))
                if sequenced:
                    prev = None
                    for p in parts:
                        steps.append(Step(f"S{sid}", p))
                        prev = p
                        sid += 1
                else:
                    for p in parts:
                        steps.append(Step(f"S{sid}", p))
                        sid += 1

        if not steps:
            steps = [Step("S1", q)]

        return Plan(original_question=q, used_llm=used_llm, ordered_steps=steps)

def split_query_simple(question: str) -> Dict[str, any]:
    splitter = QuestionSplitter()
    plan = splitter.plan(question)
    return {
        "original_question": plan.original_question,
        "used_llm": plan.used_llm,
        "steps": [s.text for s in plan.ordered_steps],
    }