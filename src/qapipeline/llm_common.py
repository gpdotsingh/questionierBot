from __future__ import annotations

import json
import re
from typing import Optional
from openai import OpenAI  
from .settings import get_provider_runtime


class LLMJsonMixin:
    CLEAN_WS = re.compile(r"\s+")

    @classmethod
    def norm(cls, s: str) -> str:
        return cls.CLEAN_WS.sub(" ", s or "").strip()

    def ask_json(self, prompt: str) -> Optional[dict]:

        try:
            if self.openai is not None:  # <- use the property
                resp = self.openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                text = resp.choices[0].message.content or ""
            elif self._ollama:
                resp = self._ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1},
                )
                text = (resp.get("message") or {}).get("content") or str(resp)
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
    _openai: Optional[OpenAI] = None          # <- not Any
    openai_model: str = ""
    _ollama: Optional[object] = None
    ollama_model: str = ""
    provider: str = ""

    def __init__(self, runtime_name: str) -> None:
        runtime = get_provider_runtime(runtime_name)
        self.provider = runtime.provider
        self._openai = runtime.openai_client
        self.openai_model = runtime.openai_model
        self._ollama = runtime.ollama_client
        self.ollama_model = runtime.ollama_model
    @property
    def openai(self) -> Optional[OpenAI]:
        return self._openai