from __future__ import annotations
import os, sys
from pathlib import Path
import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from collections import deque
from typing import Deque, Tuple
SESSIONS: Dict[str, Deque[Tuple[str, str]]] = {}

# Make src importable
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qapipeline.settings import ensure_env_loaded
ensure_env_loaded()

from qapipeline import (
    QuestionSplitter, Orchestrator, LLMCompiler, Validator,
    OrchestratorOutput, ValidatorInput
)

app = FastAPI(title="QA Pipeline Chat (Dummy Chain)", version="1.0")

class ChatRequest(BaseModel):
    message: str
    try_llm: bool = True  # splitter may ignore if no LLM configured
    session_id: str = "default"  # NEW


class ChatResponse(BaseModel):
    used_llm_in_splitter: bool
    plan_steps: List[str]
    validation_score: Optional[float] = None
    chain_trace: List[str]
    validated_response: Optional[Dict[str, Any]] = None


def _sort_key(key: str) -> int:
    if isinstance(key, str) and key.startswith("Q") and key[1:].isdigit():
        return int(key[1:])
    return 0


def _flatten_plan_steps(ordered_steps: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(ordered_steps, dict):
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

    for k, v in sorted(ordered_steps.items(), key=lambda kv: _sort_key(kv[0])):
        walk(str(k), v)
    return steps

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cwd": str(os.getcwd()),
        "have_data_file": (HERE / "data" / "CRM_Donor_Simulation_Dataset.csv").exists(),
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    sid = req.session_id
    history = SESSIONS.setdefault(sid, deque(maxlen=10))
    history.append(("user", req.message))
    memory_ctx = "\n".join(f"{role.upper()}: {content}" for role, content in history)

    # 1. Split question
    splitter = QuestionSplitter(try_llm=req.try_llm)
    plan = splitter.plan(req.message, memory_text=memory_ctx)  # NEW: pass memory
    plan_steps = _flatten_plan_steps(plan.ordered_steps)
    trace: List[str] = [f"[SPLITTER] steps={len(plan_steps)}"]

    # 2. Orchestrate (dummy executes each step)
    orch = Orchestrator(debug=True)
    answers = orch.run(plan, memory_text=memory_ctx)  # NEW: pass memory
    trace.append(f"[ORCH] produced {len(answers.query_result)} interim answers")

    # 3. Compile
    compiler = LLMCompiler()
    compiled = compiler.compile(answers)
    trace.append("[COMPILER] combined answers")

    # 4. Validate   
    validator = Validator()
    verdict = validator.validate(ValidatorInput(
        original_question=plan.original_question,
        compiled_answer=compiled.final_answer,
        plan=plan,
        orchestrator_output=answers,
        compiler_output=compiled,
    ))
    trace.append(f"[VALIDATOR] score={verdict.score:.3f}")

    final_answer = json.dumps(verdict.response_json) if verdict.response_json else compiled.final_answer

    return ChatResponse(
        used_llm_in_splitter=plan.used_llm,
        plan_steps=plan_steps,
        validation_score=verdict.score,
        chain_trace=trace,
        validated_response=verdict.response_json,
    )
