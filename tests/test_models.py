"""Tests for dataset model validation."""

import unittest

from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    IndexedChunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    PersistedIndex,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


class ModelTests(unittest.TestCase):
    """Validate the subject data models."""

    def test_dataset_accepts_answered_and_unanswered_questions(self) -> None:
        """RagDataset should accept both question variants."""

        dataset = RagDataset(
            rag_questions=[
                UnansweredQuestion(question_id="q1", question="How do I index data?"),
                AnsweredQuestion(
                    question_id="q2",
                    question="How do I search data?",
                    sources=[],
                    answer="Use the search command.",
                ),
            ]
        )

        self.assertEqual(len(dataset.rag_questions), 2)

    def test_search_and_answer_models_round_trip(self) -> None:
        """Search and answer payloads should serialize and validate as JSON."""

        source = MinimalSource(
            file_path="data/raw/example.py",
            first_character_index=2,
            last_character_index=12,
        )
        search_result = MinimalSearchResults(
            question_id="q1",
            question="What does this do?",
            retrieved_sources=[source],
        )
        student_results = StudentSearchResults(search_results=[search_result], k=5)
        answer = MinimalAnswer(**search_result.model_dump(), answer="It loads data.")
        answer_results = StudentSearchResultsAndAnswer(
            search_results=[answer],
            k=5,
        )

        restored_search = StudentSearchResults.model_validate_json(
            student_results.model_dump_json()
        )
        restored_answer = StudentSearchResultsAndAnswer.model_validate_json(
            answer_results.model_dump_json()
        )

        self.assertEqual(restored_search, student_results)
        self.assertEqual(restored_answer, answer_results)

    def test_persisted_index_validates_indexed_chunks(self) -> None:
        """Persisted index models should contain chunk and token metadata."""

        index = PersistedIndex(
            max_chunk_size=2000,
            chunk_count=1,
            document_frequencies={"api": 1},
            chunks=[
                IndexedChunk(
                    file_path="data/raw/example.py",
                    first_character_index=0,
                    last_character_index=3,
                    text="API",
                    token_counts={"api": 1},
                )
            ],
        )

        self.assertEqual(index.chunks[0].token_counts["api"], 1)

    def test_models_reject_invalid_ranges_and_control_values(self) -> None:
        """Models should reject malformed source ranges and negative values."""

        with self.assertRaises(ValidationError):
            MinimalSource(file_path="example.py", first_character_index=5, last_character_index=2)
        with self.assertRaises(ValidationError):
            StudentSearchResults(search_results=[], k=-1)
        with self.assertRaises(ValidationError):
            PersistedIndex(
                max_chunk_size=0,
                chunk_count=0,
                document_frequencies={},
                chunks=[],
            )


if __name__ == "__main__":
    unittest.main()
