"""Tests for dataset model validation."""

import unittest

from src.models import AnsweredQuestion, RagDataset, UnansweredQuestion


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


if __name__ == "__main__":
    unittest.main()
