# 🧙‍♂️ SciWizard

<p align="center"> <img src="icon/icon.ico" alt="SciWizard Icon" width="120"/> </p>
<p align="center"> <img src="gallery/image.png" alt="SciWizard Screenshot" width="800"/> </p>

---

## Features

| Area | What you get |
|------|-------------|
| **Data** | CSV loading, table preview, data profiling, missing value handling |
| **Preprocessing** | Label/one-hot encoding, column dropping, scaler info |
| **Visualisation** | Histograms, scatter plots, correlation heatmaps, feature distributions, PCA 2D |
| **Training** | 7 classification + 7 regression algorithms, hyperparameter control, CV scores |
| **AutoML** | Automatic model sweep with sortable leaderboard |
| **Evaluation** | Confusion matrix, ROC curve, cross-validation bar chart, metrics dashboard |
| **Prediction** | Form-based single prediction, batch CSV prediction with export |
| **Registry** | Persistent model save/load/delete with metadata |
| **Experiments** | JSONL-backed run history with full metric tracking |
| **Plugins** | Drop Python files into `/plugins` to add custom models and preprocessors |

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone
git clone https://github.com/pro-grammer-SD/sciwizard.git
cd sciwizard

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install
pip install -e ".[dev]"
```

---

## Usage

```bash
python -m sciwizard
```

Or use the installed entry point:

```bash
sciwizard
```

### Basic workflow

1. **Data tab** — load a CSV, select target column, handle missing values
2. **Preprocess tab** — encode categoricals, drop irrelevant columns
3. **Visualize tab** — explore distributions and correlations
4. **Train tab** — pick algorithm, configure split, train
5. **Evaluate tab** — inspect confusion matrix, ROC, CV scores
6. **Predict tab** — enter values for single prediction or upload a batch CSV
7. **Registry tab** — save and reload trained models
8. **Experiments tab** — review all past runs

---

## Writing a Plugin

Drop a `.py` file into the `plugins/` directory:

```python
# plugins/extra_trees.py
from sklearn.ensemble import ExtraTreesClassifier

def register(registry: dict) -> None:
    registry["models"]["Extra Trees"] = ExtraTreesClassifier(n_estimators=100)
```

SciWizard discovers it on next launch and adds it to the model selector.

---

## Running Tests

```bash
pytest
```

---

## Project Structure

```
sciwizard/
├── sciwizard/
│   ├── app.py                  # Bootstrap & main()
│   ├── config.py               # Constants & paths
│   ├── core/
│   │   ├── data_manager.py     # CSV loading, profiling, cleaning
│   │   ├── model_trainer.py    # Training, evaluation, AutoML
│   │   ├── model_registry.py   # Persistent model storage
│   │   ├── experiment_tracker.py
│   │   └── plugin_loader.py
│   └── ui/
│       ├── main_window.py      # Top-level window + sidebar
│       ├── theme.py            # Dark stylesheet
│       ├── workers.py          # QThread/QRunnable wrappers
│       ├── panels/             # One file per application tab
│       └── widgets/            # Shared UI components
├── tests/
├── docs/
├── plugins/                    # Drop custom model plugins here
└── icon/
    └── icon.ico
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
