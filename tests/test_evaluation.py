"""Tests for recall@k evaluation."""

import unittest

from src.evaluation import compute_recall_at_k
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


class EvaluationTests(unittest.TestCase):
    """Validate recall@k overlap handling."""

    def test_recall_counts_overlapping_spans(self) -> None:
        """Overlapping spans in the same file should count as hits."""

        answered = [
            AnsweredQuestion(
                question_id="q1",
                question="Where is the API server documented?",
                sources=[
                    MinimalSource(
                        file_path="data/raw/example.md",
                        first_character_index=10,
                        last_character_index=110,
                    )
                ],
                answer="In the example docs.",
            )
        ]
        student = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="q1",
                    question="Where is the API server documented?",
                    retrieved_sources=[
                        MinimalSource(
                            file_path="data/raw/example.md",
                            first_character_index=0,
                            last_character_index=120,
                        )
                    ],
                )
            ],
            k=5,
        )

        self.assertEqual(compute_recall_at_k(student, answered, 1), 1.0)


if __name__ == "__main__":
    unittest.main()
