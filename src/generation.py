"""Grounded answer generation using the local Qwen model."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .models import MinimalSource, PersistedIndex

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARACTERS = 12000


def build_context(
    index: PersistedIndex,
    sources: list[MinimalSource],
    max_characters: int = MAX_CONTEXT_CHARACTERS,
) -> str:
    """Collect retrieved chunk text in ranked order for the model prompt."""

    if max_characters <= 0:
        return ""

    chunks_by_location = {
        (
            chunk.file_path,
            chunk.first_character_index,
            chunk.last_character_index,
        ): chunk.text
        for chunk in index.chunks
    }
    context_parts: list[str] = []
    characters_used = 0
    for source in sources:
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index,
        )
        text = chunks_by_location.get(key)
        if text is None:
            continue
        remaining = max_characters - characters_used
        if remaining <= 0:
            break
        context_parts.append(f"[{source.file_path}]\n{text[:remaining]}")
        characters_used += min(len(text), remaining)
    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    """Build a prompt that restricts the answer to retrieved source context."""

    return (
        "Answer the question using only the source context below. "
        "If the context does not contain the answer, say so clearly. "
        "Do not invent facts or cite sources that are not provided.\n\n"
        f"Question:\n{question}\n\n"
        f"Source context:\n{context}"
    )


def generate_answer(
    question: str,
    sources: list[MinimalSource],
    index: PersistedIndex | None,
    model_name: str = MODEL_NAME,
) -> str:
    """Generate a grounded answer from retrieved chunks using Qwen."""

    if not question.strip():
        return "Empty questions cannot be answered."
    if index is None:
        return "No index is available to generate a grounded answer."

    context = build_context(index, sources)
    if not context:
        return "No retrieved source context was available for this question."

    prompt = build_prompt(question, context)
    try:
        tokenizer, model, torch = _load_model(model_name)
        messages = [
            {
                "role": "system",
                "content": "You are a concise assistant answering questions about a codebase.",
            },
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return answer or "The model did not generate an answer from the retrieved context."
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return (
            "The local Qwen generator is unavailable. Install the model dependencies "
            f"and ensure {model_name} can be loaded. Retrieved context was found "
            f"but generation failed: {error}"
        )


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> tuple[Any, Any, Any]:
    """Load and cache the tokenizer, model, and torch module lazily."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, torch


def build_placeholder_answer(question: str, sources: list[MinimalSource]) -> str:
    """Return a compatibility fallback for callers without a persisted index."""

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
