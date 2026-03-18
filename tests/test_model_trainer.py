"""Unit tests for ModelTrainer."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from sciwizard.core.model_trainer import ModelTrainer, TrainingResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data[["mean radius", "mean texture", "mean perimeter", "mean area"]]
    y = pd.Series(data.target, name="target")
    return X, y


@pytest.fixture
def regression_data():
    data = load_diabetes(as_frame=True)
    X = data.data[["age", "bmi", "bp"]]
    y = data.target
    return X, y


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classification_returns_result(classification_data):
    X, y = classification_data
    trainer = ModelTrainer(task_type="classification", test_size=0.2, random_state=42)
    result = trainer.train("Logistic Regression", X, y)

    assert isinstance(result, TrainingResult)
    assert result.task_type == "classification"
    assert "Accuracy" in result.metrics
    assert 0.0 <= result.metrics["Accuracy"] <= 1.0
    assert result.train_duration_s >= 0


def test_classification_all_models(classification_data):
    from sciwizard.core.model_trainer import CLASSIFICATION_MODELS

    X, y = classification_data
    trainer = ModelTrainer(task_type="classification", test_size=0.3, random_state=0)

    for name in CLASSIFICATION_MODELS:
        result = trainer.train(name, X, y)
        assert result.metrics["Accuracy"] >= 0.0, f"{name} returned negative accuracy"


def test_classification_cv_scores(classification_data):
    X, y = classification_data
    trainer = ModelTrainer(task_type="classification")
    result = trainer.train("Random Forest", X, y)
    assert result.cv_scores.size == 5
    assert all(0 <= s <= 1 for s in result.cv_scores)


def test_classification_predictions_shape(classification_data):
    X, y = classification_data
    trainer = ModelTrainer(task_type="classification", test_size=0.2)
    result = trainer.train("Decision Tree", X, y)
    assert len(result.y_pred) == len(result.y_test)


def test_classification_unknown_model(classification_data):
    X, y = classification_data
    trainer = ModelTrainer(task_type="classification")
    with pytest.raises(ValueError, match="Unknown model"):
        trainer.train("Super Fancy Model 9000", X, y)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_regression_returns_result(regression_data):
    X, y = regression_data
    trainer = ModelTrainer(task_type="regression", test_size=0.2, random_state=42)
    result = trainer.train("Linear Regression", X, y)

    assert isinstance(result, TrainingResult)
    assert result.task_type == "regression"
    assert "R²" in result.metrics
    assert "MAE" in result.metrics
    assert "RMSE" in result.metrics


def test_regression_r2_in_range(regression_data):
    X, y = regression_data
    trainer = ModelTrainer(task_type="regression")
    result = trainer.train("Random Forest", X, y)
    assert result.metrics["R²"] <= 1.0


# ---------------------------------------------------------------------------
# AutoML
# ---------------------------------------------------------------------------


def test_automl_returns_sorted_leaderboard(classification_data):
    X, y = classification_data
    trainer = ModelTrainer(task_type="classification")
    leaderboard = trainer.automl(X, y)

    assert len(leaderboard) > 0
    scores = [e.score for e in leaderboard]
    assert scores == sorted(scores, reverse=True), "Leaderboard not sorted by score"


def test_automl_all_entries_have_names(classification_data):
    X, y = classification_data
    trainer = ModelTrainer(task_type="classification")
    leaderboard = trainer.automl(X, y)
    for entry in leaderboard:
        assert entry.model_name
        assert entry.metric


# ---------------------------------------------------------------------------
# Invalid task type
# ---------------------------------------------------------------------------


def test_invalid_task_type():
    with pytest.raises(ValueError, match="task_type must be"):
        ModelTrainer(task_type="clustering")
