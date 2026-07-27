"""Chunking helpers for Python and Markdown/text files."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A chunk of source text with its character span."""

    start: int
    end: int
    text: str


def chunk_content(content: str, suffix: str, max_chunk_size: int) -> list[TextChunk]:
    """Split content into bounded chunks based on file type."""

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")
    if not content:
        return []
    if suffix == ".py":
        return _chunk_python(content, max_chunk_size)
    return _chunk_text(content, max_chunk_size)


def _chunk_python(content: str, max_chunk_size: int) -> list[TextChunk]:
    """Split Python source around top-level declarations when possible."""

    blocks: list[TextChunk] = []
    current_start = 0
    lines = content.splitlines(keepends=True)
    offset = 0

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        is_boundary = (
            index > 0
            and not line.startswith((" ", "\t"))
            and stripped.startswith(("def ", "class ", "async def "))
        )
        if is_boundary and offset > current_start:
            blocks.extend(_split_large_chunk(content, current_start, offset, max_chunk_size))
            current_start = offset
        offset += len(line)

    if current_start < len(content):
        blocks.extend(_split_large_chunk(content, current_start, len(content), max_chunk_size))
    return blocks


def _chunk_text(content: str, max_chunk_size: int) -> list[TextChunk]:
    """Split Markdown/text content around paragraph boundaries when possible."""

    separators = ["\n\n", "\n#", "\n##", "\n###"]
    chunks: list[TextChunk] = []
    start = 0

    while start < len(content):
        stop = min(start + max_chunk_size, len(content))
        boundary = stop
        if stop < len(content):
            for separator in separators:
                candidate = content.rfind(separator, start, stop)
                if candidate > start:
                    boundary = candidate + len(separator)
                    break
        if boundary <= start:
            boundary = stop
        chunks.append(TextChunk(start=start, end=boundary, text=content[start:boundary]))
        start = boundary
    return chunks


def _split_large_chunk(
    content: str,
    start: int,
    end: int,
    max_chunk_size: int,
) -> list[TextChunk]:
    """Split an oversized chunk into bounded pieces."""

    chunks: list[TextChunk] = []
    current = start
    while current < end:
        stop = min(current + max_chunk_size, end)
        chunks.append(TextChunk(start=current, end=stop, text=content[current:stop]))
        current = stop
    return chunks
