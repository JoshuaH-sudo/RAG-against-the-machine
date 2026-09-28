"""Index building and lexical retrieval helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_content
from .io_utils import ensure_directory
from .models import IndexedChunk, MinimalSource, PersistedIndex

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}


def build_index(
    raw_directory: Path,
    output_directory: Path,
    max_chunk_size: int,
    repository_root: Path,
) -> PersistedIndex:
    """Build and persist a lexical index for supported files."""

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")
    ensure_directory(output_directory)
    indexed_chunks: list[IndexedChunk] = []
    document_frequencies: Counter[str] = Counter()

    files = sorted(
        (
            path
            for path in raw_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
        ),
        key=lambda path: path.as_posix(),
    )

    for path in tqdm(files, desc="Indexing", unit="file"):
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            relative_path = path.relative_to(repository_root).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        for chunk in chunk_content(content, path.suffix.lower(), max_chunk_size):
            token_counts = Counter(_tokenize(chunk.text))
            if not token_counts:
                continue
            document_frequencies.update(token_counts.keys())
            indexed_chunks.append(
                IndexedChunk(
                    file_path=relative_path,
                    first_character_index=chunk.start,
                    last_character_index=chunk.end,
                    text=chunk.text,
                    token_counts=dict(token_counts),
                )
            )

    persisted_index = PersistedIndex(
        max_chunk_size=max_chunk_size,
        chunk_count=len(indexed_chunks),
        document_frequencies=dict(document_frequencies),
        chunks=indexed_chunks,
    )
    index_path = output_directory / "index.json"
    index_path.write_text(
        persisted_index.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return persisted_index


def load_index(index_directory: Path) -> PersistedIndex:
    """Load a persisted lexical index from disk."""

    index_path = index_directory / "index.json"
    return PersistedIndex.model_validate_json(index_path.read_text(encoding="utf-8"))


def search_index(index: PersistedIndex, query: str, k: int) -> list[MinimalSource]:
    """Run a lightweight TF-IDF-style retrieval over indexed chunks."""

    if not query.strip() or k <= 0:
        return []
    query_terms = Counter(_tokenize(query))
    if not query_terms:
        return []

    total_chunks = max(index.chunk_count, 1)
    scored_sources: list[tuple[float, MinimalSource]] = []

    for chunk in index.chunks:
        score = _score_chunk(
            chunk.token_counts,
            query_terms,
            index.document_frequencies,
            total_chunks,
        )
        if score <= 0:
            continue
        scored_sources.append(
            (
                score,
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                ),
            )
        )

    scored_sources.sort(
        key=lambda item: (
            -item[0],
            item[1].file_path,
            item[1].first_character_index,
        )
    )
    return [source for _, source in scored_sources[:k]]


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase identifiers and words."""

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _score_chunk(
    chunk_counts: dict[str, int],
    query_terms: Counter[str],
    document_frequencies: dict[str, int],
    total_chunks: int,
) -> float:
    """Compute a simple TF-IDF score for a chunk."""

    score = 0.0
    for term, query_count in query_terms.items():
        term_frequency = chunk_counts.get(term, 0)
        if term_frequency == 0:
            continue
        doc_frequency = document_frequencies.get(term, 0)
        idf = math.log((1 + total_chunks) / (1 + doc_frequency)) + 1.0
        score += (1.0 + math.log(term_frequency)) * (1.0 + math.log(query_count)) * idf
    return score
