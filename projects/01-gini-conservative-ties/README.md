# VS01 — Gini: Conservative Handling of Ties

This mini-project supports my Validation Series carousel on **Gini** behaviour when model scores contain **ties**.

## Why this exists
When predicted scores have ties, some implementations become sensitive to **row order** (especially CAP-based constructions).
For model validation and reporting, this is a reliability problem.

This project provides:
- **AUC-based Gini (standard)**: ties treated neutrally as 0.5 (sklearn ROC AUC).
- **AUC-based Gini (conservative)**: subtract a **permutation-invariant** correction based on tie groups.
- **CAP-based Gini** with explicit tie-breaking: `conservative / stable / optimistic` (order-sensitive by design).

## API
Inputs:
- `y` / `y_true`: binary labels in `{0,1}` (1 = positive/bad class)
- `s` / `y_score`: model scores (float), higher = more positive/bad

```python
from validation_series_quant.vs01_gini_ties.metrics import (
    gini_auc,
    gini_cap,
    conservative_ties_correction,
)
```

## Where to look
- Core implementation: `src/validation_series_quant/vs01_gini_ties/metrics.py`
- Tests: `tests/test_gini_ties.py`
- Demo notebook: `projects/01-gini-conservative-ties/notebooks/01_demo_ties.ipynb`

## How to run
Install:
```bash
poetry install
```

Tests:
```bash
poetry run pytest
```

Demo notebook:
- Open `notebooks/01_demo_ties.ipynb`
- Restart kernel & Run all

## Expected results (sanity)
With ties present:
- `gini_cap(..., tie_break="stable")` can vary under row permutations
- `conservative_ties_correction(...)` is permutation-invariant
- `gini_cap(conservative) <= gini_cap(stable) <= gini_cap(optimistic)`
- `gini_auc(conservative_ties=True) <= gini_auc(conservative_ties=False)`

## Links
- LinkedIn: https://www.linkedin.com/in/alexey-mengden
