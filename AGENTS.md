# Repository Guidelines

## Project Structure & Modules
- Core plugin lives in `beetsplug/bitrater`: `plugin.py` wires into beets, `analyzer.py` orchestrates the analysis pipeline, and helpers such as `classifier.py`, `cutoff_detector.py`, `transcode_detector.py`, and `spectrum.py` handle feature extraction and scoring.
- Shared types and constants are in `beetsplug/bitrater/types.py` and `constants.py`. Reference training assets under `beetsplug/bitrater/training_data`.
- Tests reside in `tests/` (unit and integration). Long-form design notes sit in `docs/plans/`. Images/assets live in `images/` and data samples in `documents/` or `tests/data`.

## Build, Test, and Development Commands
- Install dev tools: `uv sync` (uses `uv.lock`) or `pip install -e ".[dev]"`.
- Run the suite: `uv run pytest -v` (or plain `pytest` if the env is active).
- Lint/format: `uv run ruff check .`, `uv run black .`, `uv run isort .`.
- Package/build: `uv build` (hatchling backend). Beets command entry point during manual checks: `beet bitrater [--train --threads N --verbose]`.

## Coding Style & Naming Conventions
- Python 3.10+ with type hints; keep line length ≤100 (Black/Ruff config). Prefer snake_case for functions/variables, PascalCase for classes, and lower_snake_case module filenames.
- Follow existing logging patterns via the module-level `logger`; avoid bare prints.
- Apply automated tools (Black, Isort, Ruff/Flake8) before submitting; keep imports sorted and unused symbols removed.

## Testing Guidelines
- Pytest is configured in `pyproject.toml`; tests match `test_*.py` with verbose output and short tracebacks.
- Add unit coverage near the component (e.g., new spectrum logic → `tests/test_spectrum.py`). Integration scenarios belong in `tests/test_integration.py`.
- Include fixtures or sample audio under `tests/data/`; keep heavy assets out of the repo.
- Aim to cover edge cases (low-confidence scores, thread safety, transcode detection boundaries) and note any skips or assumptions.

## Commit & Pull Request Guidelines
- Use concise, imperative commits (e.g., `Add V2 to VBR presets`, `test: add integration coverage`). Keep scope focused; include a prefix when clarifying the area (`test:`, `docs:`, `fix:`).
- PRs should summarize changes, rationale, and risks; link issues when applicable and list manual/automated test results. Provide configuration notes (model paths, training dirs) if reviewers must reproduce locally.
- Keep diffs minimal, typed, and linted; prefer incremental PRs over monolithic changes.

## Data, Models, and Configuration Notes
- Classifier models are optional; point `model_path` in the config if you load a saved model. Training data directories can be swapped via `training_dir` or CLI flags.
- Avoid committing large binaries or generated models. Document any non-default settings used during testing or profiling so they can be repeated.***
