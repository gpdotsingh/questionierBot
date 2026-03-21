from __future__ import annotations
import os, sys
from pathlib import Path
import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Header
from pydantic import BaseModel
from collections import deque
from typing import Deque, Tuple
from fastapi.middleware.cors import CORSMiddleware

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
    ValidatorInput
)
from observability import start_trace, end_trace, flush

app = FastAPI(title="QA Pipeline Chat (Dummy Chain)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    try_llm: bool = True  # splitter may ignore if no LLM configured
    session_id: str = "default"


class ChatResponse(BaseModel):
    used_llm_in_splitter: bool
    plan_steps: List[str]
    validation_score: Optional[float] = None
    chain_trace: List[str]
    validated_response: Optional[Dict[str, Any]] = None
    # Set when the QB access token was silently refreshed mid-request.
    # The frontend should store this as the new JWT for subsequent calls.
    refreshed_jwt: Optional[str] = None


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


def _truncate_words(text: Any, max_words: int = 1000) -> Any:
    if not isinstance(text, str):
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ... [truncated]"


def _truncate_value(val: Any, max_words: int = 1000) -> Any:
    if isinstance(val, list):
        return [_truncate_value(v, max_words) for v in val]
    if isinstance(val, dict):
        return {k: _truncate_value(v, max_words) for k, v in val.items()}
    return _truncate_words(val, max_words)


def _truncate_response_json(data: Any, max_words: int = 1000) -> Any:
    if not isinstance(data, dict):
        return data
    truncated = dict(data)
    citations = []
    for cite in data.get("citations", []):
        if isinstance(cite, dict):
            citations.append({
                **cite,
                "query": _truncate_words(cite.get("query"), max_words),
                "output": _truncate_value(cite.get("output"), max_words),
            })
        else:
            citations.append(cite)
    truncated["citations"] = citations
    return truncated


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cwd": str(os.getcwd()),
        "have_quickbooks_metadata": (HERE / "metadata" / "quickbooks_data.yaml").exists(),
        "have_quickbooks_semantics": (HERE / "data" / "quickbooks_semantics.yaml").exists(),
    }

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    authorization: Optional[str] = Header(default=None),
) -> ChatResponse:
    sid = req.session_id
    history = SESSIONS.setdefault(sid, deque(maxlen=10))
    history.append(("user", req.message))
    memory_ctx = "\n".join(f"{role.upper()}: {content}" for role, content in history)

    # Langfuse trace (no-op when not configured)
    start_trace(
        name="chat",
        user_id=sid,
        session_id=sid,
        metadata={"message": req.message},
    )

    # 1. Split question
    splitter = QuestionSplitter(try_llm=req.try_llm)
    plan = splitter.plan(req.message, memory_text=memory_ctx)
    plan_steps = _flatten_plan_steps(plan.ordered_steps)
    trace: List[str] = [f"[SPLITTER] steps={len(plan_steps)}"]

    # 2. Orchestrate — forward the JWT from the frontend so Spring Boot is called
    #    with the user's own token instead of the static env-var token.
    orch = Orchestrator(debug=True, jwt_token=authorization)
    answers = orch.run(plan, memory_text=memory_ctx)
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

    truncated_response = _truncate_response_json(verdict.response_json, max_words=1000)
    final_answer = json.dumps(truncated_response) if truncated_response else compiled.final_answer

    # Close Langfuse trace
    end_trace(output=compiled.final_answer, score=verdict.score)
    flush()

    return ChatResponse(
        used_llm_in_splitter=plan.used_llm,
        plan_steps=plan_steps,
        validation_score=verdict.score,
        chain_trace=trace,
        validated_response=truncated_response,
        refreshed_jwt=orch.refreshed_jwt,
    )
