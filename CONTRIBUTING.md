# Contributing to SciWizard

Thanks for your interest. Here's how to contribute effectively.

---

## Setup

```bash
git clone https://github.com/pro-grammer-SD/sciwizard.git
cd sciwizard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Coding Standards

- Python 3.10+, type hints on every function signature
- Google-style docstrings on all public methods and classes
- Line length ≤ 100 characters (`ruff` enforces this)
- No global mutable state
- All heavy work runs in a `Worker`/`LongWorker` — never block the Qt event loop
- Log with `logging.getLogger(__name__)`, not `print()`

Run linting before committing:

```bash
ruff check sciwizard/
```

---

## Architecture Rules

| Layer | Allowed dependencies |
|-------|----------------------|
| `core/` | stdlib, numpy, pandas, sklearn, joblib |
| `ui/` | `core/`, PySide6, matplotlib |
| `tests/` | `core/`, pytest |

`core/` must never import from `ui/`. Keep business logic and presentation fully separated.

---

## Adding a New Panel

1. Create `sciwizard/ui/panels/my_panel.py` subclassing `QWidget`
2. Add it to `MainWindow._build_ui()` and the `_NAV_ITEMS` list
3. Connect cross-panel signals via `MainWindow._connect_signals()`

---

## Adding a New Algorithm

Add it to `CLASSIFICATION_MODELS` or `REGRESSION_MODELS` in `core/model_trainer.py`.
Alternatively, use the plugin system for external contributions.

---

## PR Workflow

1. Fork and create a feature branch: `git checkout -b feat/my-feature`
2. Write or update tests for your change
3. Run `pytest` — all tests must pass
4. Open a PR against `main` using the PR template
5. At least one approval is required to merge

---

## Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add k-fold stratified split option
fix: handle empty dataframe in profiler
docs: update plugin guide
refactor: extract metric rendering into MetricCard widget
```
