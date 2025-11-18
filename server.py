from __future__ import annotations
import os, sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# Make src importable
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# (Optional) load .env (does NOT forward provider/model explicitly)
try:
    from dotenv import load_dotenv, find_dotenv
    env_file = (HERE / ".env")
    if env_file.exists():
        load_dotenv(env_file)
    else:
        found = find_dotenv()
        if found:
            load_dotenv(found)
except Exception:
    pass

from qapipeline import (
    QuestionSplitter, Orchestrator, LLMCompiler, Validator,
    CompilerInput, ValidatorInput
)

app = FastAPI(title="QA Pipeline Chat (Dummy Chain)", version="1.0")

class ChatRequest(BaseModel):
    message: str
    try_llm: bool = True  # splitter may ignore if no LLM configured

class ChatResponse(BaseModel):
    answer: str
    used_llm_in_splitter: bool
    plan_steps: List[str]
    validation_score: Optional[float] = None
    chain_trace: List[str]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cwd": str(os.getcwd()),
        "have_data_file": (HERE / "data" / "CRM_Donor_Simulation_Dataset.csv").exists(),
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # 1. Split question
    splitter = QuestionSplitter(try_llm=req.try_llm)
    plan = splitter.plan(req.message)
    trace: List[str] = [f"[SPLITTER] steps={len(plan.ordered_steps)}"]

    # 2. Orchestrate (dummy executes each step)
    orch = Orchestrator(debug=True)
    answers = orch.run(plan)
    trace.append(f"[ORCH] produced {len(answers)} interim answers")

    # 3. Compile
    compiler = LLMCompiler()
    compiled = compiler.compile(CompilerInput(
        original_question=plan.original_question,
        answers=answers
    ))
    trace.append("[COMPILER] combined answers")

    # 4. Validate
    validator = Validator()
    verdict = validator.validate(ValidatorInput(
        original_question=plan.original_question,
        compiled_answer=compiled.final_answer
    ))
    trace.append(f"[VALIDATOR] score={verdict.score:.3f}")

    return ChatResponse(
        answer=compiled.final_answer,
        used_llm_in_splitter=plan.used_llm,
        plan_steps=[f"{s.id}: {s.text}" for s in plan.ordered_steps],
        validation_score=verdict.score,
        chain_trace=trace,
    )