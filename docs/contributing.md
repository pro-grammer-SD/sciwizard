# Contributing

This mirrors [CONTRIBUTING.md](../CONTRIBUTING.md) with additional technical detail for first-time contributors.

---

## Development setup

```bash
git clone https://github.com/pro-grammer-SD/sciwizard.git
cd sciwizard

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Verify everything works:

```bash
pytest            # all tests pass
ruff check sciwizard/   # no lint errors
python -m sciwizard     # GUI launches
```

---

## Repository layout

```
sciwizard/       ← main package (see architecture.md for detail)
tests/           ← pytest unit tests; no Qt required for core tests
docs/            ← markdown documentation
plugins/         ← example and user-contributed plugins
icon/            ← application icon (icon.ico)
.github/         ← CI workflow, issue templates, PR template
```

---

## Making a change

**Bug fix or small improvement:**

1. Create a branch: `git checkout -b fix/describe-the-bug`
2. Write a failing test that reproduces the bug
3. Fix the code until the test passes
4. Open a PR

**New feature:**

1. Open an issue first to discuss scope and approach
2. Create a branch: `git checkout -b feat/my-feature`
3. Implement with tests
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Open a PR referencing the issue

---

## Coding rules (non-negotiable)

- Type hints on every function parameter and return type
- Google-style docstrings on every public class and method
- `core/` must not import from `ui/` — ever
- Heavy work goes in a `Worker` or `LongWorker` — never block the Qt event loop
- Use `logging.getLogger(__name__)` — no bare `print()` in production code
- Constants and paths go in `config.py` — no magic strings scattered around

---

## Tests

Core tests (`test_data_manager.py`, `test_model_trainer.py`, etc.) do not require a display. Run them anywhere.

Qt UI tests require a display or `QT_QPA_PLATFORM=offscreen`. In CI this is handled by xvfb.

```bash
pytest tests/ -v
pytest tests/test_data_manager.py -v   # single file
pytest -k "test_load"                  # by name pattern
```

---

## Commit messages

```
feat: short description of new behaviour
fix: short description of what broke and how it's fixed
docs: what was updated
refactor: what was restructured (no behaviour change)
test: add or update tests
chore: dependency bumps, CI config, tooling
```

Keep the first line under 72 characters. Add a body paragraph when the "why" is non-obvious.
