"""Tests for grounded answer-generation helpers."""

import unittest

from src.generation import build_context, build_prompt, generate_answer
from src.models import IndexedChunk, MinimalSource, PersistedIndex


class GenerationTests(unittest.TestCase):
    """Validate context assembly and generation fallbacks."""

    def setUp(self) -> None:
        """Create a small persisted index for generation tests."""

        self.index = PersistedIndex(
            max_chunk_size=2000,
            chunk_count=1,
            document_frequencies={"api": 1},
            chunks=[
                IndexedChunk(
                    file_path="data/raw/example.md",
                    first_character_index=0,
                    last_character_index=25,
                    text="The API uses token auth.",
                    token_counts={"the": 1, "api": 1, "uses": 1, "token": 1, "auth": 1},
                )
            ],
        )
        self.source = MinimalSource(
            file_path="data/raw/example.md",
            first_character_index=0,
            last_character_index=25,
        )

    def test_context_and_prompt_include_ranked_source(self) -> None:
        """The prompt should include the retrieved source and question."""

        context = build_context(self.index, [self.source])
        prompt = build_prompt("How does authentication work?", context)

        self.assertIn("data/raw/example.md", context)
        self.assertIn("The API uses token auth.", context)
        self.assertIn("How does authentication work?", prompt)

    def test_generation_falls_back_without_an_index(self) -> None:
        """Generation should remain safe when retrieval data is unavailable."""

        answer = generate_answer("What is this?", [self.source], None)

        self.assertIn("No index is available", answer)

    def test_generation_handles_empty_question(self) -> None:
        """Generation should reject an empty question without loading a model."""

        answer = generate_answer("", [self.source], self.index)

        self.assertEqual(answer, "Empty questions cannot be answered.")


if __name__ == "__main__":
    unittest.main()
