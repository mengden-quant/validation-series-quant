# validation-series-quant

[![CI](https://github.com/mengden-quant/validation-series-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/mengden-quant/validation-series-quant/actions/workflows/ci.yml)

Supporting code for my LinkedIn Validation Series on quantitative finance.

## What’s inside
This repo contains small, reproducible mini-projects:
- clean Python package layout (`src/`)
- tests (`pytest`)
- linting (`ruff`)
- CI via GitHub Actions

## Projects

### VS01 — Gini: Conservative Handling of Ties
- Problem: score ties can make some Gini implementations sensitive to row order.
- Implementation: AUC-based conservative correction (permutation-invariant) + CAP tie-breaking demo.
- Project README: `projects/01-gini-conservative-ties/README.md`
- Demo notebook: `projects/01-gini-conservative-ties/notebooks/01_demo_ties.ipynb`

## Quickstart
Install:
```bash
poetry install
```

Run quality checks (same as CI):
```bash
poetry run ruff check .
poetry run pytest
```

## Repo structure
- `src/validation_series_quant/` — package code
- `tests/` — unit tests
- `projects/` — per-carousel mini-projects (docs + notebooks)

## Links
- LinkedIn: https://www.linkedin.com/in/alexey-mengden
