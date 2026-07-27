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
3. Lexical retrieval foundation (TF-IDF or BM25)
4. Honest, grounded answer generation with local models
5. Recall@k measurement and reproducible outputs

## Suggested agent workflow
1. Read `README.md` and the current subject notes.
2. Run `uv sync` before development if dependencies are missing.
3. Use `make lint` and `make test` after changing Python code.
4. Update `README.md` whenever architecture or commands change.
5. Keep this file and `.github/workflows/copilot-setup-steps.yml` aligned.
