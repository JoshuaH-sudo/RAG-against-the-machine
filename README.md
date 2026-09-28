*This project has been created as part of the 42 curriculum by jhoban.*

# RAG-against-the-machine

## Description
This repository contains a **basic project skeleton** for the 42 Berlin **RAG against the machine** subject. The goal of the final project is to index a codebase, retrieve the most relevant source snippets for a question, and generate grounded answers from that retrieved context.

The project includes the mandatory repository layout, pydantic data models, a Python Fire CLI, lexical indexing and retrieval, lazy local Qwen answer generation, and local evaluation helpers.

## Instructions
### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

### Installation
```bash
make install
```

### Core commands
```bash
uv run python -m src index --max_chunk_size=2000
uv run python -m src search "How is the OpenAI server configured?" --k=5
uv run python -m src search_dataset \
  --dataset_path=data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k=10 \
  --save_directory=data/output/search_results/UnansweredQuestions
uv run python -m src answer "How is the OpenAI server configured?" --k=5
uv run python -m src answer_dataset \
  --student_search_results_path=data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory=data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path=data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path=data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### Development helpers
```bash
make lint
make test
make clean
```

## System architecture
The scaffold follows the mandatory pipeline split:
1. **Indexing** (`src/indexing.py`) reads text-like files from `data/raw/`, chunks them, tokenizes them, and persists a lexical index under `data/processed/`.
2. **Retrieval** (`src/indexing.py`) loads the persisted index and ranks chunks with a lightweight TF-IDF-style scorer.
3. **Answer generation** (`src/generation.py`) reconstructs retrieved chunks into a bounded prompt and lazily loads `Qwen/Qwen3-0.6B` through Transformers when the answer commands are used. If model dependencies or weights are unavailable, it returns an explicit grounded-generation error instead of inventing an answer.
4. **Evaluation** (`src/evaluation.py`) computes recall@k with the same same-file plus overlapping-span rule described in the subject.
5. **CLI orchestration** (`src/cli.py`) exposes all required subject commands through Python Fire.

## Chunking strategy
Two chunking paths are included:
- **Python files** are split around top-level `def`, `class`, and `async def` boundaries when possible, with hard fallback splitting for oversized blocks.
- **Markdown/text files** are split around paragraph and heading boundaries when possible, again with a bounded fallback split.

All chunks are capped by `--max_chunk_size`, matching the subject’s 2000-character retrieval limit.

## Retrieval method
The scaffold implements a simple lexical retriever built around:
- lowercased identifier/word tokenization,
- per-chunk term counts,
- corpus-level document frequencies,
- TF-IDF-style ranking for top-k retrieval.

This gives the repository a working baseline that can later be replaced or extended with BM25, hybrid retrieval, caching, or semantic embeddings.

## Performance analysis
This project is designed for correctness, structure, and graceful error handling first. It is **not yet tuned** to hit the target recall@5 or throughput thresholds from the subject. Those measurements should be recorded after indexing the supplied corpus and running the official evaluation workflow.

## Design decisions
- **uv** is used as the package manager to match the subject and evaluator expectations.
- **pydantic** defines the required exchange formats to keep file I/O and CLI payloads validated.
- **Python Fire** provides the mandatory CLI surface with minimal boilerplate.
- **tqdm** is used in dataset and indexing loops so long-running commands show progress.
- The answer generator is intentionally explicit about being a scaffold instead of pretending to be a final model-backed implementation.

## Challenges faced
The main challenge in a skeleton-first repository is balancing minimalism with usefulness. Instead of shipping empty command stubs, this scaffold includes just enough real indexing, retrieval, JSON output, and evaluation logic to make future iterations concrete while keeping the implementation small and easy to explain during a review.

## Example usage
A minimal local smoke test flow looks like this:
```bash
uv run python -m src index --max_chunk_size=500
uv run python -m src search "tokenization" --k=3
uv run python -m src answer "tokenization" --k=3
```

## Resources
- 42 subject PDF: *RAG against the machine*
- Python Fire: https://google.github.io/python-fire/
- Pydantic: https://docs.pydantic.dev/
- tqdm: https://tqdm.github.io/
- uv: https://docs.astral.sh/uv/
- Introduction to TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf

### How AI was used
AI was used to help translate the subject requirements into a repository scaffold, including the initial CLI layout, data-model wiring, documentation structure, and agent working guidelines. The generated code and docs were then reviewed, adjusted, and validated locally so the repository remains understandable and easy to extend manually.
