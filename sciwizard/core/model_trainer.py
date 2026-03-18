"""Model training, evaluation, and AutoML logic."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sciwizard.config import CV_FOLDS, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalogue of available models
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=DEFAULT_RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE),
    "K-Nearest Neighbours": KNeighborsClassifier(),
    "SVM (RBF)": SVC(probability=True, random_state=DEFAULT_RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
}

REGRESSION_MODELS: dict[str, Any] = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=DEFAULT_RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=DEFAULT_RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(random_state=DEFAULT_RANDOM_STATE),
    "K-Nearest Neighbours": KNeighborsRegressor(),
    "SVR": SVR(),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Holds all artefacts produced by a single training run."""

    model_name: str
    task_type: str  # "classification" | "regression"
    pipeline: Pipeline
    label_encoder: LabelEncoder | None
    feature_names: list[str]
    classes: list[Any] | None

    # Split
    X_train: pd.DataFrame = field(repr=False)
    X_test: pd.DataFrame = field(repr=False)
    y_train: pd.Series = field(repr=False)
    y_test: pd.Series = field(repr=False)
    y_pred: np.ndarray = field(repr=False)

    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)
    cv_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    train_duration_s: float = 0.0

    # For ROC
    y_prob: np.ndarray | None = field(default=None, repr=False)


@dataclass
class AutoMLEntry:
    """A single row in the AutoML leaderboard."""

    model_name: str
    score: float
    metric: str
    cv_mean: float
    cv_std: float
    duration_s: float


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ModelTrainer:
    """Orchestrates training, evaluation, and AutoML sweeps.

    Args:
        task_type: Either ``"classification"`` or ``"regression"``.
        test_size: Fraction of data reserved for evaluation.
        random_state: Seed for reproducibility.
        scale_features: Whether to add a StandardScaler to the pipeline.
    """

    def __init__(
        self,
        task_type: str = "classification",
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        scale_features: bool = True,
    ) -> None:
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparams: dict | None = None,
    ) -> TrainingResult:
        """Train a single named model.

        Args:
            model_name: Key from CLASSIFICATION_MODELS or REGRESSION_MODELS.
            X: Feature matrix (numeric).
            y: Target vector.
            hyperparams: Optional dict of hyperparameters to set on the estimator.

        Returns:
            Populated TrainingResult.
        """
        catalogue = (
            CLASSIFICATION_MODELS
            if self.task_type == "classification"
            else REGRESSION_MODELS
        )
        if model_name not in catalogue:
            raise ValueError(f"Unknown model: {model_name!r}")

        import copy

        estimator = copy.deepcopy(catalogue[model_name])
        if hyperparams:
            estimator.set_params(**hyperparams)

        return self._run_training(model_name, estimator, X, y)

    def automl(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        progress_callback: Any | None = None,
    ) -> list[AutoMLEntry]:
        """Try every model in the catalogue and return a sorted leaderboard.

        Args:
            X: Feature matrix.
            y: Target vector.
            progress_callback: Optional callable(int, int) for progress updates.

        Returns:
            List of AutoMLEntry sorted by score descending.
        """
        catalogue = (
            CLASSIFICATION_MODELS
            if self.task_type == "classification"
            else REGRESSION_MODELS
        )
        leaderboard: list[AutoMLEntry] = []
        total = len(catalogue)

        for idx, (name, estimator) in enumerate(catalogue.items()):
            import copy

            try:
                t0 = time.perf_counter()
                Xs, ys, _ = self._prepare_data(X, y)
                pipeline = self._build_pipeline(copy.deepcopy(estimator))
                scoring = (
                    "accuracy" if self.task_type == "classification" else "r2"
                )
                scores = cross_val_score(
                    pipeline, Xs, ys, cv=CV_FOLDS, scoring=scoring
                )
                duration = time.perf_counter() - t0
                leaderboard.append(
                    AutoMLEntry(
                        model_name=name,
                        score=float(scores.mean()),
                        metric=scoring,
                        cv_mean=float(scores.mean()),
                        cv_std=float(scores.std()),
                        duration_s=round(duration, 3),
                    )
                )
                logger.info("AutoML %s → %.4f", name, scores.mean())
            except Exception as exc:
                logger.warning("AutoML skipped %s: %s", name, exc)

            if progress_callback:
                progress_callback(idx + 1, total)

        leaderboard.sort(key=lambda e: e.score, reverse=True)
        return leaderboard

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, LabelEncoder | None]:
        """Encode categoricals in X and optionally encode y."""
        X = X.copy()
        le = None

        # Encode non-numeric columns (catches both object and pandas StringDtype)
        for col in X.select_dtypes(include=["object", "category", "string"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        if self.task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=y.name)

        return X, y, le

    def _build_pipeline(self, estimator: Any) -> Pipeline:
        steps: list = []
        if self.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)

    def _run_training(
        self,
        model_name: str,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> TrainingResult:
        feature_names = list(X.columns)
        Xp, yp, le = self._prepare_data(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            Xp,
            yp,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        pipeline = self._build_pipeline(estimator)

        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        duration = time.perf_counter() - t0

        y_pred = pipeline.predict(X_test)
        y_prob = None
        classes = None

        if self.task_type == "classification":
            try:
                classes = list(pipeline.named_steps["model"].classes_)
            except (AttributeError, KeyError):
                classes = list(np.unique(y_train))
            metrics = self._classification_metrics(y_test, y_pred)
            if hasattr(pipeline, "predict_proba"):
                with contextlib.suppress(Exception):
                    y_prob = pipeline.predict_proba(X_test)
        else:
            metrics = self._regression_metrics(y_test, y_pred)

        scoring = "accuracy" if self.task_type == "classification" else "r2"
        cv_scores = cross_val_score(pipeline, Xp, yp, cv=CV_FOLDS, scoring=scoring)

        return TrainingResult(
            model_name=model_name,
            task_type=self.task_type,
            pipeline=pipeline,
            label_encoder=le,
            feature_names=feature_names,
            classes=classes,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics=metrics,
            cv_scores=cv_scores,
            train_duration_s=round(duration, 4),
        )

    @staticmethod
    def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        avg = "weighted"
        return {
            "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "Precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "F1 Score": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        }

    @staticmethod
    def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "R²": round(float(r2_score(y_true, y_pred)), 4),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        }
