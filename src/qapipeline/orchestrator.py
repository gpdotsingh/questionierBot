from __future__ import annotations
from typing import List
from .models import Plan
from .settings import get_provider_runtime, ensure_env_loaded
import re

# ---------- Optional FAISS store (cosine) ----------
try:
    from src.ingestdata.faiss_store import FaissVectorStoreCosine
except Exception:
    FaissVectorStoreCosine = None

class Orchestrator:
    """
    Dummy orchestrator: for each step produces a synthetic answer string.
    Input: Plan
    Output: List[str] (answers) passed to compiler.
    """
    def __init__(self, debug: bool = False, **kwargs):
        ensure_env_loaded()
        self.debug = debug
        self.runtime = get_provider_runtime("orchestrator")

    def run(self, plan: Plan) -> List[str]:
        provider = self.runtime.provider
        self.provider = provider
        answers: List[str] = []
        cumulative = []
