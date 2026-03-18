"""Experiment tracking: log every training run to a JSONL file."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from sciwizard.config import EXPERIMENT_LOG_PATH
from sciwizard.core.model_trainer import TrainingResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Append training run metadata to a JSONL log file.

    Each line in the log represents one experiment run. The log is human-
    readable and easy to import into pandas.

    Args:
        log_path: Override the default log file location.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path or EXPERIMENT_LOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        result: TrainingResult,
        dataset_name: str = "unknown",
        model_id: str | None = None,
        notes: str = "",
    ) -> None:
        """Append one experiment run to the log.

        Args:
            result: Completed TrainingResult.
            dataset_name: Name of the dataset used.
            model_id: Registry ID if the model was saved.
            notes: Free-text annotation.
        """
        entry = {
            "run_id": self._new_run_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": dataset_name,
            "model_name": result.model_name,
            "task_type": result.task_type,
            "metrics": result.metrics,
            "cv_mean": float(result.cv_scores.mean()) if result.cv_scores.size else None,
            "cv_std": float(result.cv_scores.std()) if result.cv_scores.size else None,
            "train_duration_s": result.train_duration_s,
            "model_id": model_id,
            "notes": notes,
            "n_features": len(result.feature_names),
            "n_train": len(result.X_train),
            "n_test": len(result.X_test),
        }
        with self._path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        logger.info("Experiment logged: %s (run %s)", result.model_name, entry["run_id"])

    def load_history(self) -> list[dict]:
        """Read all past experiments from the log file.

        Returns:
            List of experiment dicts, newest first.
        """
        if not self._path.exists():
            return []
        entries = []
        with self._path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping corrupt log line")
        return list(reversed(entries))

    def clear(self) -> None:
        """Wipe the experiment log."""
        if self._path.exists():
            self._path.write_text("")
        logger.info("Experiment log cleared")

    @staticmethod
    def _new_run_id() -> str:
        import uuid

        return str(uuid.uuid4())[:8]
