"""Metric formatting and interpretation helpers."""

from __future__ import annotations

# Maps task type → ordered list of primary metric names
CLASSIFICATION_METRIC_ORDER = ["Accuracy", "F1 Score", "Precision", "Recall"]
REGRESSION_METRIC_ORDER = ["R²", "RMSE", "MAE"]


def primary_metric(task_type: str, metrics: dict[str, float]) -> tuple[str, float]:
    """Return the most important metric name and value for a given task.

    Args:
        task_type: ``"classification"`` or ``"regression"``.
        metrics: Dict of metric names to values.

    Returns:
        Tuple of (metric_name, value).
    """
    order = (
        CLASSIFICATION_METRIC_ORDER
        if task_type == "classification"
        else REGRESSION_METRIC_ORDER
    )
    for name in order:
        if name in metrics:
            return name, metrics[name]
    # Fallback: first available
    if metrics:
        k, v = next(iter(metrics.items()))
        return k, v
    return "—", 0.0


def format_metric(name: str, value: float) -> str:
    """Format a metric name/value pair for display.

    Args:
        name: Metric name.
        value: Numeric value.

    Returns:
        Formatted string, e.g. ``"Accuracy: 0.9412"``.
    """
    return f"{name}: {value:.4f}"


def score_colour(value: float, task_type: str = "classification") -> str:
    """Return a hex colour indicating whether a score is good, mediocre, or poor.

    For classification this uses the 0–1 accuracy/F1 scale.
    For regression (R²) the same 0–1 scale is used; negative R² is always red.

    Args:
        value: The metric score.
        task_type: ``"classification"`` or ``"regression"``.

    Returns:
        Hex colour string.
    """
    if value < 0:
        return "#f38ba8"   # red — negative R²
    if value >= 0.9:
        return "#a6e3a1"   # green
    if value >= 0.7:
        return "#f9e2af"   # yellow
    return "#f38ba8"       # red
