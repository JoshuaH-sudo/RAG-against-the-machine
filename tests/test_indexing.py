"""Tests for lexical indexing and retrieval."""

import unittest

from src.indexing import search_index
from src.models import IndexedChunk, PersistedIndex


class IndexingTests(unittest.TestCase):
    """Validate the TF-IDF lexical retriever."""

    def test_tfidf_ranks_matching_chunk_first(self) -> None:
        """A distinctive query term should rank its matching chunk first."""

        index = PersistedIndex(
            max_chunk_size=2000,
            chunk_count=2,
            document_frequencies={"api": 2, "authentication": 1, "database": 1},
            chunks=[
                IndexedChunk(
                    file_path="docs/api.md",
                    first_character_index=0,
                    last_character_index=40,
                    text="The API uses token authentication.",
                    token_counts={"the": 1, "api": 1, "uses": 1, "token": 1, "authentication": 1},
                ),
                IndexedChunk(
                    file_path="docs/database.md",
                    first_character_index=0,
                    last_character_index=35,
                    text="The API connects to the database.",
                    token_counts={"the": 2, "api": 1, "connects": 1, "to": 1, "database": 1},
                ),
            ],
        )

        results = search_index(index, "token authentication", k=1)

        self.assertEqual(results[0].file_path, "docs/api.md")

    def test_empty_or_unknown_query_returns_no_results(self) -> None:
        """Queries without indexed terms should not return arbitrary chunks."""

        index = PersistedIndex(
            max_chunk_size=2000,
            chunk_count=1,
            document_frequencies={"api": 1},
            chunks=[
                IndexedChunk(
                    file_path="docs/api.md",
                    first_character_index=0,
                    last_character_index=15,
                    text="The API works.",
                    token_counts={"the": 1, "api": 1, "works": 1},
                )
            ],
        )

        self.assertEqual(search_index(index, "", k=5), [])
        self.assertEqual(search_index(index, "unrelated", k=5), [])


if __name__ == "__main__":
    unittest.main()
