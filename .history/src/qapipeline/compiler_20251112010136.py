from __future__ import annotations
from .models import CompilerInput, CompilerOutput

class LLMCompiler:
    """
    Dummy compiler: concatenate answers into a final paragraph.
    Receives original question (prev input) + answers (prev output).
    """
    def compile(self, data: CompilerInput) -> CompilerOutput:
        joined = "\n".join(data.answers)
        final = f"Question: {data.original_question}\nSynthesized Answer:\n{joined}"
        return CompilerOutput(final_answer=final, details={"answer_count": len(data.answers)})