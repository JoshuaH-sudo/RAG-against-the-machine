"""Tests for lexical indexing and retrieval."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.indexing import build_index, search_index
from src.models import IndexedChunk, PersistedIndex


class IndexingTests(unittest.TestCase):
    """Validate the TF-IDF lexical retriever."""

    def test_build_index_persists_bounded_chunks_and_statistics(self) -> None:
        """Indexing should persist exact paths, offsets, tokens, and frequencies."""

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_directory = root / "data" / "raw"
            output_directory = root / "data" / "processed"
            raw_directory.mkdir(parents=True)
            (raw_directory / "module.py").write_text(
                "class Example:\n    pass\n\ndef run():\n    return Example()\n",
                encoding="utf-8",
            )
            (raw_directory / "guide.md").write_text(
                "# Guide\n\nUse the example API.\n\n# Details\n\nMore API details.",
                encoding="utf-8",
            )

            persisted_index = build_index(
                raw_directory,
                output_directory,
                max_chunk_size=30,
                repository_root=root,
            )

            self.assertEqual(persisted_index.chunk_count, len(persisted_index.chunks))
            self.assertTrue((output_directory / "index.json").exists())
            self.assertIn("api", persisted_index.document_frequencies)
            self.assertTrue(
                all(
                    len(chunk.text) <= 30
                    and chunk.last_character_index - chunk.first_character_index <= 30
                    for chunk in persisted_index.chunks
                )
            )
            self.assertEqual(
                {chunk.file_path for chunk in persisted_index.chunks},
                {"data/raw/guide.md", "data/raw/module.py"},
            )

    def test_build_index_rejects_non_positive_chunk_size(self) -> None:
        """Indexing should reject invalid chunk sizes before scanning files."""

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(ValueError):
                build_index(root, root / "processed", 0, root)

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

    def test_retrieval_limits_results_and_breaks_ties_by_source_location(self) -> None:
        """Retrieval should return at most k results in a stable order."""

        index = PersistedIndex(
            max_chunk_size=2000,
            chunk_count=3,
            document_frequencies={"api": 3},
            chunks=[
                IndexedChunk(
                    file_path="z-last.py",
                    first_character_index=0,
                    last_character_index=10,
                    text="api",
                    token_counts={"api": 1},
                ),
                IndexedChunk(
                    file_path="a-first.py",
                    first_character_index=20,
                    last_character_index=30,
                    text="api",
                    token_counts={"api": 1},
                ),
                IndexedChunk(
                    file_path="a-first.py",
                    first_character_index=0,
                    last_character_index=10,
                    text="api",
                    token_counts={"api": 1},
                ),
            ],
        )

        results = search_index(index, "api", k=2)

        self.assertEqual(
            [(source.file_path, source.first_character_index) for source in results],
            [("a-first.py", 0), ("a-first.py", 20)],
        )


if __name__ == "__main__":
    unittest.main()
