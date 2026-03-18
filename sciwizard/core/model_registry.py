"""Model registry: persistent save/load with metadata and versioning."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from sciwizard.config import MODEL_REGISTRY_DIR
from sciwizard.core.model_trainer import TrainingResult

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Save, load, list, and delete trained models on disk.

    Each model is stored as a directory under MODEL_REGISTRY_DIR::

        ~/.sciwizard/models/<model_id>/
            model.joblib   — serialised sklearn Pipeline
            meta.json      — metadata (metrics, features, timestamps, …)
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self._dir = registry_dir or MODEL_REGISTRY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, result: TrainingResult, alias: str | None = None) -> str:
        """Persist a trained model and return its unique model_id.

        Args:
            result: The TrainingResult from ModelTrainer.train().
            alias: Optional human-readable name. Defaults to model_name.

        Returns:
            Unique model_id (UUID string).
        """
        model_id = str(uuid.uuid4())[:8]
        model_dir = self._dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Serialise pipeline
        joblib.dump(result.pipeline, model_dir / "model.joblib")

        # Build metadata
        meta: dict[str, Any] = {
            "model_id": model_id,
            "alias": alias or result.model_name,
            "model_name": result.model_name,
            "task_type": result.task_type,
            "feature_names": result.feature_names,
            "classes": [str(c) for c in result.classes] if result.classes else None,
            "metrics": result.metrics,
            "cv_mean": float(result.cv_scores.mean()) if result.cv_scores.size else None,
            "cv_std": float(result.cv_scores.std()) if result.cv_scores.size else None,
            "train_duration_s": result.train_duration_s,
            "saved_at": datetime.utcnow().isoformat(),
        }
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        logger.info("Saved model %s → %s", result.model_name, model_id)
        return model_id

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, model_id: str) -> tuple[Any, dict]:
        """Load a model pipeline and its metadata.

        Args:
            model_id: The id returned by save().

        Returns:
            Tuple of (pipeline, metadata_dict).

        Raises:
            FileNotFoundError: If model_id does not exist.
        """
        model_dir = self._dir / model_id
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_id!r} not found in registry")

        pipeline = joblib.load(model_dir / "model.joblib")
        meta = json.loads((model_dir / "meta.json").read_text())
        logger.info("Loaded model %s (%s)", model_id, meta.get("model_name"))
        return pipeline, meta

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return all saved models sorted newest-first.

        Returns:
            List of metadata dicts.
        """
        entries = []
        for model_dir in sorted(self._dir.iterdir(), reverse=True):
            meta_path = model_dir / "meta.json"
            if meta_path.exists():
                try:
                    entries.append(json.loads(meta_path.read_text()))
                except json.JSONDecodeError:
                    logger.warning("Corrupt meta.json in %s — skipping", model_dir)
        return entries

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, model_id: str) -> None:
        """Remove a model from the registry.

        Args:
            model_id: The id to delete.
        """
        import shutil

        model_dir = self._dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info("Deleted model %s", model_id)
        else:
            logger.warning("Delete requested for unknown model_id %s", model_id)
