"""Answer generation helpers for the scaffold implementation."""

from __future__ import annotations

from .models import MinimalSource


def build_placeholder_answer(question: str, sources: list[MinimalSource]) -> str:
    """Return an honest placeholder answer for the scaffold."""

    if not question.strip():
        return "Empty questions cannot be answered."
    if not sources:
        return (
            "No sources were retrieved for this question. Run indexing first or refine "
            "the query before wiring in the local Qwen/Qwen3-0.6B generator."
        )
    source_list = ", ".join(source.file_path for source in sources[:3])
    return (
        "This scaffold retrieved relevant context from "
        f"{source_list}. Replace the placeholder generator in src/generation.py "
        "with a local Qwen/Qwen3-0.6B integration to produce grounded natural-language answers."
    )
