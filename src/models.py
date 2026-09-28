"""Pydantic models for the RAG pipeline."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MinimalSource(BaseModel):
    """Represents one retrieved source span."""

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_span(self) -> "MinimalSource":
        """Ensure source offsets describe a forward character span."""

        if self.last_character_index < self.first_character_index:
            raise ValueError("last_character_index must not precede first_character_index")
        return self


class UnansweredQuestion(BaseModel):
    """Represents a question without an answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question with sources and an answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represents search results for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents a search result paired with an answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represents a batch of search results."""

    search_results: list[MinimalSearchResults]
    k: int = Field(ge=0)


class StudentSearchResultsAndAnswer(BaseModel):
    """Represents a batch of answers."""

    search_results: list[MinimalAnswer]
    k: int = Field(ge=0)


class IndexedChunk(BaseModel):
    """Represents one indexed chunk stored on disk."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    token_counts: dict[str, int]


class PersistedIndex(BaseModel):
    """Represents the persisted lexical index."""

    max_chunk_size: int = Field(gt=0)
    chunk_count: int = Field(ge=0)
    document_frequencies: dict[str, int]
    chunks: list[IndexedChunk]


class JsonMessage(BaseModel):
    """Represents a CLI status payload."""

    status: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
