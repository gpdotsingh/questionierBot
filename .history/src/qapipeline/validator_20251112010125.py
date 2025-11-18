from __future__ import annotations
from .models import ValidatorInput, ValidatorOutput

class Validator:
    """
    Dummy validator: score is a min(1.0, chars / 800).
    """
    def validate(self, data: ValidatorInput) -> ValidatorOutput:
        length = len(data.compiled_answer)
        score = min(1.0, length / 800.0)
        return ValidatorOutput(score=score, notes=f"Length-based heuristic ({length} chars)")