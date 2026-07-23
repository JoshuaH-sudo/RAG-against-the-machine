"""Python Fire CLI for the project scaffold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from fire import Fire
from pydantic import ValidationError
from tqdm import tqdm

from .evaluation import compute_recall_at_k
from .generation import build_placeholder_answer
from .indexing import build_index, load_index, search_index
from .io_utils import load_json, save_json
from .models import (
    AnsweredQuestion,
    JsonMessage,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


class RagCli:
    """Command line interface for the RAG scaffold."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_directory: str = "data/raw",
        output_directory: str = "data/processed",
    ) -> str:
        """Index the corpus and persist the lexical search data."""

        repository_root = Path.cwd()
        raw_path = repository_root / raw_directory
        output_path = repository_root / output_directory

        if max_chunk_size <= 0:
            return self._emit_message("error", "max_chunk_size must be greater than zero")
        if not raw_path.exists():
            return self._emit_message("error", f"Missing raw directory: {raw_path}")

        persisted_index = build_index(raw_path, output_path, max_chunk_size, repository_root)
        return self._emit_message(
            "ok",
            f"Indexed {persisted_index.chunk_count} chunks into {output_path.as_posix()}",
            {
                "max_chunk_size": max_chunk_size,
                "chunk_count": persisted_index.chunk_count,
            },
        )

    def search(
        self,
        query: str,
        k: int = 5,
        index_directory: str = "data/processed",
    ) -> str:
        """Search the index for a single query."""

        question = UnansweredQuestion(question=query)
        return self._single_search(question, k, Path.cwd() / index_directory)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        index_directory: str = "data/processed",
    ) -> str:
        """Search an entire dataset and save StudentSearchResults JSON."""

        if k < 0:
            return self._emit_message("error", "k must be zero or greater")
        try:
            dataset = self._load_dataset(Path.cwd() / dataset_path)
        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as error:
            return self._emit_message("error", f"Failed to load dataset: {error}")

        index_path = Path.cwd() / index_directory
        search_results: list[MinimalSearchResults] = []
        for item in tqdm(dataset.rag_questions, desc="Searching", unit="question"):
            question = self._to_unanswered_question(item)
            payload = self._single_search_payload(question, k, index_path)
            search_results.append(payload)

        student_results = StudentSearchResults(search_results=search_results, k=k)
        destination = Path.cwd() / save_directory / Path(dataset_path).name
        save_json(destination, student_results.model_dump())
        return self._emit_message(
            "ok",
            f"Saved student_search_results to {destination.as_posix()}",
            {"question_count": len(search_results), "k": k},
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        index_directory: str = "data/processed",
    ) -> str:
        """Answer a single query using retrieved context."""

        question = UnansweredQuestion(question=query)
        search_payload = self._single_search_payload(question, k, Path.cwd() / index_directory)
        answer = build_placeholder_answer(search_payload.question, search_payload.retrieved_sources)
        result = MinimalAnswer(**search_payload.model_dump(), answer=answer)
        return json.dumps(result.model_dump(), indent=2)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> str:
        """Generate placeholder answers from saved search results."""

        source_path = Path.cwd() / student_search_results_path
        try:
            payload = StudentSearchResults.model_validate(load_json(source_path))
        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as error:
            return self._emit_message(
                "error",
                f"Failed to load student search results: {error}",
            )

        answers: list[MinimalAnswer] = []
        for item in tqdm(payload.search_results, desc="Answering", unit="question"):
            answer = build_placeholder_answer(item.question, item.retrieved_sources)
            answers.append(MinimalAnswer(**item.model_dump(), answer=answer))

        result = StudentSearchResultsAndAnswer(search_results=answers, k=payload.k)
        destination = Path.cwd() / save_directory / source_path.name
        save_json(destination, result.model_dump())
        return self._emit_message(
            "ok",
            f"Saved student_search_results_and_answer to {destination.as_posix()}",
            {"question_count": len(answers), "k": payload.k},
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> str:
        """Compute recall@k against an answered dataset."""

        try:
            student_results = StudentSearchResults.model_validate(
                load_json(Path.cwd() / student_search_results_path)
            )
            dataset = self._load_dataset(Path.cwd() / dataset_path)
        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as error:
            return self._emit_message("error", f"Failed to load evaluation inputs: {error}")

        answered_questions = [
            item for item in dataset.rag_questions if isinstance(item, AnsweredQuestion)
        ]
        recall_summary = {
            f"recall@{value}": round(
                compute_recall_at_k(student_results, answered_questions, value),
                4,
            )
            for value in _metric_points(student_results.k)
        }
        return self._emit_message(
            "ok",
            "Evaluation complete",
            {"question_count": len(answered_questions), **recall_summary},
        )

    def _single_search(self, question: UnansweredQuestion, k: int, index_path: Path) -> str:
        """Run a single search and return a JSON string."""

        payload = self._single_search_payload(question, k, index_path)
        return json.dumps(payload.model_dump(), indent=2)

    def _single_search_payload(
        self,
        question: UnansweredQuestion,
        k: int,
        index_path: Path,
    ) -> MinimalSearchResults:
        """Run a single search and return the model payload."""

        if k < 0:
            return MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[],
            )
        if not question.question.strip() or k == 0:
            return MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[],
            )
        try:
            index = load_index(index_path)
        except FileNotFoundError:
            return MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[],
            )
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=search_index(index, question.question, k),
        )

    def _load_dataset(self, dataset_path: Path) -> RagDataset:
        """Load and validate a dataset file."""

        return RagDataset.model_validate(load_json(dataset_path))

    def _emit_message(
        self,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Serialize a status message as JSON."""

        payload = JsonMessage(status=status, message=message, details=details or {})
        return payload.model_dump_json(indent=2)

    def _to_unanswered_question(
        self,
        question: AnsweredQuestion | UnansweredQuestion,
    ) -> UnansweredQuestion:
        """Convert answered questions to unanswered ones for retrieval."""

        if isinstance(question, UnansweredQuestion) and not isinstance(
            question, AnsweredQuestion
        ):
            return question
        answered_question = cast(AnsweredQuestion, question)
        return UnansweredQuestion(
            question_id=answered_question.question_id,
            question=answered_question.question,
        )


def _metric_points(k: int) -> list[int]:
    """Return the recall@k values to report."""

    candidates = [1, 3, 5, k]
    return sorted({value for value in candidates if value > 0})


def main() -> None:
    """Run the Fire CLI."""

    Fire(RagCli)
