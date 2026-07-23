PYTHON := uv run python -m src

.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	$(PYTHON) --help

debug:
	uv run python -m pdb src/__main__.py -- --help

clean:
	find . -type d \( -name '__pycache__' -o -name '.mypy_cache' -o -name '.pytest_cache' \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run python -m unittest discover -s tests -p 'test_*.py'
