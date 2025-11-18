from __future__ import annotations
from typing import List
from .models import Plan

class Orchestrator:
    """
    Dummy orchestrator: for each step produces a synthetic answer string.
    Input: Plan
    Output: List[str] (answers) passed to compiler.
    """
    def __init__(self, debug: bool = False, **kwargs):
        self.debug = debug

    def run(self, plan: Plan) -> List[str]:
        answers: List[str] = []
        cumulative = []
        for step in plan.ordered_steps:
            cumulative.append(step.text)
            ans = f"[{step.id}] processed -> {step.text}"
            if self.debug:
                ans += f" | context_so_far={len(cumulative)}"
            answers.append(ans)
        return answers