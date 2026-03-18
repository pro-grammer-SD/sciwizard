## 📋 Summary

<!-- One clear paragraph. What does this PR do and why? -->

---

## 🔖 Type of change

- [ ] 🐛 Bug fix — non-breaking, resolves an existing issue
- [ ] ✨ New feature — non-breaking, adds new behaviour
- [ ] 💥 Breaking change — alters existing behaviour or public API
- [ ] ♻️ Refactor — internal restructuring, no behaviour change
- [ ] 📚 Documentation only
- [ ] 🔧 CI / tooling / dependency update

---

## 🔗 Related issues

Closes #<!-- issue number -->

---

## 🧩 Changes made

<!-- Bullet list: what changed, in which file/component, and why -->

- `sciwizard/...` —
- `tests/...` —

---

## 🏗️ Architecture check

- [ ] `core/` does **not** import from `ui/`
- [ ] No new global mutable state introduced
- [ ] All heavy operations run in a `Worker` or `LongWorker` — UI thread is never blocked
- [ ] Constants and paths added to `config.py`, not scattered inline

---

## 🧪 Testing

- [ ] New or updated unit tests added in `tests/`
- [ ] All existing tests pass — `pytest`
- [ ] Lint passes — `ruff check sciwizard/`
- [ ] Manually verified in the running GUI

<!-- Describe the manual test scenario if applicable -->

---

## 🎨 UI / UX (if applicable)

- [ ] Follows the existing dark theme and QSS conventions
- [ ] No hardcoded pixel sizes or colours outside `theme.py` / `config.py`
- [ ] Tooltips added for any new ML concepts (Beginner Mode)

**Screenshots** — before / after:

| Before | After |
|--------|-------|
|        |       |

---

## 📝 Code quality

- [ ] Type hints on every function signature
- [ ] Google-style docstrings on all public classes and methods
- [ ] No bare `print()` statements — logging used throughout
- [ ] No leftover debug code or commented-out blocks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

---

## 🔌 Plugin system (if adding a model or preprocessor)

- [ ] Registered via `register(registry)` in `plugins/`
- [ ] Does not modify `CLASSIFICATION_MODELS` or `REGRESSION_MODELS` directly
- [ ] Example plugin file or docs updated if relevant

---

## ⚠️ Breaking changes (if applicable)

<!-- Describe what breaks, why the change is necessary, and the migration path -->

---

## 🗒️ Additional notes

<!-- Anything reviewers should know — performance impact, known edge cases, follow-up issues -->
