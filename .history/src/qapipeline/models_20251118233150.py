from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Step:
    id: str
    text: str

@dataclass
class Plan:
    original_question: str
    used_llm: bool
    ordered_steps: List[Step]

@dataclass
class CompilerInput:
    original_question: str
    answers: List[str]

@dataclass
class CompilerOutput:
    final_answer: str
    details: Optional[dict] = None

@dataclass
class ValidatorInput:
    original_question: str
    compiled_answer: str

@dataclass
class ValidatorOutput:
    score: float
    notes: Optional[str] = None