# agent.md

## Mission
- Build the project incrementally and keep every change explainable during a 42 evaluation.
- Prefer small, reviewable steps over large speculative rewrites.
- Treat this repository as a Python 3.10+ `uv` project with a CLI entrypoint at `uv run python -m src`.

## Working rules for AI agents
- Follow the 42 subject before adding features that are not explicitly required.
- Prefer standard-library solutions unless an external dependency is clearly required by the subject.
- Keep retrieved source spans at or below the configured `--max_chunk_size`.
- Never hard-code dataset or output paths; expose them through CLI arguments.
- When an agent needs the author's 42 intra, use `jhoban`.
- Handle malformed input, missing files, and empty queries without uncaught exceptions.
- Keep generated outputs, model weights, and large datasets out of Git.
- Validate changes with the smallest relevant command set before asking for review.

## Project priorities
1. Data models with `pydantic`
2. CLI commands with Python Fire
3. Lexical retrieval foundation (TF-IDF)
4. Honest, grounded answer generation with local models
5. Recall@k measurement and reproducible outputs

## Suggested agent workflow
1. Read `README.md` and the current subject notes.
2. Run `uv sync` before development if dependencies are missing.
3. Use `make lint` and `make test` after changing Python code.
4. Update `README.md` whenever architecture or commands change.
5. Keep this file and `.github/workflows/copilot-setup-steps.yml` aligned.

## Exercise implementation checklist
1. Prepare the repository as a Python 3.10+ `uv` project with `src/`,
	`pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, and `.gitignore`.
2. Index the provided vLLM corpus from `data/raw/` and preserve exact corpus
	paths such as `data/raw/vllm-0.10.1/...`.
3. Implement separate chunking strategies for Python code and Markdown/text.
	Respect the configurable `--max_chunk_size`, defaulting to 2000 characters;
	never return a source span longer than 2000 characters.
4. Persist chunk text, character offsets, token counts, and document
	frequencies under `data/processed/`.
5. Use TF-IDF as the mandatory lexical retrieval method. Tokenize queries and
	chunks consistently, rank matching chunks, and return the top `k` source
	locations. Empty, unknown, or zero-result queries must return an empty list.
6. Implement and validate the required Pydantic models for sources, questions,
	datasets, search results, and answers.
7. Implement grounded answer generation using `Qwen/Qwen3-0.6B` by default.
	Pass retrieved context within the model budget and emit validated JSON.
8. Expose every required operation through Python Fire:
	`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, and
	`evaluate`. Keep all input and output paths configurable through CLI flags.
9. Use `tqdm` for long-running indexing and batch operations. Handle missing
	files, malformed JSON, empty queries, nonsensical queries, and invalid `k`
	values without uncaught tracebacks.
10. Run the reproducible pipeline in this order:
	 `index` -> `search_dataset` -> moulinette evaluation -> `answer_dataset`.
	 Scope search and answer output directories by dataset name.
11. Evaluate retrieval against the official criteria: at least 80% recall@5 on
	 documentation questions, at least 50% recall@5 on code questions, indexing
	 in at most 5 minutes, and retrieval for 200 questions in at most 90 seconds.
12. Keep the README in English and document the architecture, chunking,
	 TF-IDF ranking, performance, design decisions, challenges, AI usage, and
	 runnable examples.
13. Add tests for chunking, indexing, retrieval, models, output formats, and
	 degenerate inputs. Run `make test` and the relevant lint checks after edits.
14. Do not implement bonus features until the mandatory pipeline is complete
	 and validated. Possible bonuses are embeddings, hybrid retrieval,
	 incremental indexing, caching, and a local HTTP API.
