# AGENTS.md

## Python

Be sure to maintain code formatting. You can use the ruff formatter/linter as needed: `uv run ruff format`,  `uv run ruff check --fix .` and `uv run ty check`.

## frontend

Use CI-friendly checks for frontend changes. Run `pnpm -C frontend exec eslint src` and ensure no React Hooks warnings.
