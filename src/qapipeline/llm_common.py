from __future__ import annotations

import json
import re
from typing import Any, Optional
from .settings import get_provider_runtime

# Optional observability — no-op when not installed or not configured
try:
    from observability import log_generation as _log_generation
except Exception:
    def _log_generation(**kwargs: Any) -> None:  # type: ignore[misc]
        pass


class LLMJsonMixin:
    CLEAN_WS = re.compile(r"\s+")

    @classmethod
    def norm(cls, s: str) -> str:
        return cls.CLEAN_WS.sub(" ", s or "").strip()

    def ask_json(self, prompt: str) -> Optional[dict]:
        try:
            if self.openai is not None:
                resp = self.openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                text = resp.choices[0].message.content or ""
                # Log to Langfuse
                usage = {}
                if hasattr(resp, "usage") and resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                _log_generation(
                    name=getattr(self, "_component_name", "unknown"),
                    model=self.openai_model,
                    prompt=prompt,
                    completion=text,
                    usage=usage,
                )
            elif self._ollama:
                resp = self._ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1},
                )
                text = (resp.get("message") or {}).get("content") or str(resp)
                # Log to Langfuse
                usage = {}
                if isinstance(resp, dict):
                    p_tokens = resp.get("prompt_eval_count", 0) or 0
                    c_tokens = resp.get("eval_count", 0) or 0
                    usage = {
                        "prompt_tokens": p_tokens,
                        "completion_tokens": c_tokens,
                        "total_tokens": p_tokens + c_tokens,
                    }
                _log_generation(
                    name=getattr(self, "_component_name", "unknown"),
                    model=self.ollama_model,
                    prompt=prompt,
                    completion=text,
                    usage=usage,
                )
            else:
                return None
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(text[s:e + 1])
            return json.loads(text)
        except Exception as e:
            print(f"ERROR: {e}")
            return None


class LLMRouterBase(LLMJsonMixin):
    _openai: Optional[Any] = None
    openai_model: str = ""
    _ollama: Optional[object] = None
    ollama_model: str = ""
    provider: str = ""
    _component_name: str = "unknown"

    def __init__(self, runtime_name: str) -> None:
        self._component_name = runtime_name.lower()
        runtime = get_provider_runtime(runtime_name)
        self.provider = runtime.provider
        self._openai = runtime.openai_client
        self.openai_model = runtime.openai_model
        self._ollama = runtime.ollama_client
        self.ollama_model = runtime.ollama_model

    @property
    def openai(self) -> Optional[Any]:
        return self._openai
