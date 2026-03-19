"""
Langfuse observability wrapper.

Provides trace/generation logging for the QA pipeline.
All functions are no-op when Langfuse env vars are not configured,
so the pipeline works identically with or without Langfuse.
"""
from __future__ import annotations

import os
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Lazy singleton — created on first use.
_langfuse_instance = None
_langfuse_checked = False

# Context variable holds the active trace for the current request.
_current_trace: ContextVar[Optional[Any]] = ContextVar("_current_trace", default=None)


def _get_langfuse() -> Optional[Any]:
    """Return the Langfuse client singleton, or None if not configured."""
    global _langfuse_instance, _langfuse_checked
    if _langfuse_checked:
        return _langfuse_instance
    _langfuse_checked = True

    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    if not secret or not public:
        _langfuse_instance = None
        return None

    try:
        from langfuse import Langfuse  # type: ignore

        _langfuse_instance = Langfuse(
            secret_key=secret,
            public_key=public,
            host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
    except Exception as exc:
        print(f"[observability] Failed to init Langfuse: {exc}")
        _langfuse_instance = None
    return _langfuse_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_trace(
    name: str = "chat",
    user_id: str = "",
    session_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Create a Langfuse trace for the current request and store it in context."""
    try:
        lf = _get_langfuse()
        if lf is None:
            return
        trace = lf.trace(
            name=name,
            user_id=user_id or None,
            session_id=session_id or None,
            metadata=metadata or {},
        )
        _current_trace.set(trace)
    except Exception as exc:
        print(f"[observability] start_trace error: {exc}")


def end_trace(
    output: str = "",
    score: Optional[float] = None,
) -> None:
    """Update the current trace with the final output and optional validation score."""
    try:
        trace = _current_trace.get(None)
        if trace is None:
            return
        trace.update(output=output)
        if score is not None:
            lf = _get_langfuse()
            if lf is not None:
                lf.score(
                    trace_id=trace.id,
                    name="validation_score",
                    value=score,
                )
    except Exception as exc:
        print(f"[observability] end_trace error: {exc}")


def log_generation(
    name: str = "llm",
    model: str = "",
    prompt: str = "",
    completion: str = "",
    usage: Optional[Dict[str, int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a single LLM generation (prompt + response + tokens) under the
    current trace.  Called from ask_json() in llm_common.py.
    """
    try:
        trace = _current_trace.get(None)
        if trace is None:
            # No active trace — Langfuse not configured or start_trace not called.
            return
        trace.generation(
            name=name,
            model=model,
            input=prompt,
            output=completion,
            usage=usage or {},
            metadata=metadata or {},
        )
    except Exception as exc:
        print(f"[observability] log_generation error: {exc}")


def flush() -> None:
    """Flush pending Langfuse events (call before returning the HTTP response)."""
    try:
        lf = _get_langfuse()
        if lf is not None:
            lf.flush()
    except Exception as exc:
        print(f"[observability] flush error: {exc}")
