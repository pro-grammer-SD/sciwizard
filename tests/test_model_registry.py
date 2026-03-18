"""Unit tests for ModelRegistry save/load/list/delete."""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.datasets import load_iris

from sciwizard.core.model_registry import ModelRegistry
from sciwizard.core.model_trainer import ModelTrainer


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(registry_dir=tmp_path / "registry")


@pytest.fixture
def training_result():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    trainer = ModelTrainer(task_type="classification", test_size=0.2, random_state=42)
    return trainer.train("Random Forest", X, y)


def test_save_returns_id(registry, training_result):
    model_id = registry.save(training_result)
    assert isinstance(model_id, str)
    assert len(model_id) > 0


def test_save_creates_files(registry, training_result):
    model_id = registry.save(training_result)
    model_dir = registry._dir / model_id
    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "meta.json").exists()


def test_load_returns_pipeline_and_meta(registry, training_result):
    model_id = registry.save(training_result)
    pipeline, meta = registry.load(model_id)

    assert pipeline is not None
    assert meta["model_name"] == "Random Forest"
    assert meta["task_type"] == "classification"
    assert "metrics" in meta


def test_load_nonexistent_raises(registry):
    with pytest.raises(FileNotFoundError):
        registry.load("doesnotexist")


def test_list_models(registry, training_result):
    assert registry.list_models() == []
    registry.save(training_result, alias="iris-rf-v1")
    registry.save(training_result, alias="iris-rf-v2")
    models = registry.list_models()
    assert len(models) == 2
    aliases = {m["alias"] for m in models}
    assert "iris-rf-v1" in aliases
    assert "iris-rf-v2" in aliases


def test_delete_removes_entry(registry, training_result):
    model_id = registry.save(training_result)
    assert len(registry.list_models()) == 1
    registry.delete(model_id)
    assert len(registry.list_models()) == 0


def test_delete_nonexistent_is_silent(registry):
    registry.delete("ghost_id")  # should not raise


def test_saved_model_can_predict(registry, training_result):
    model_id = registry.save(training_result)
    pipeline, meta = registry.load(model_id)

    X_test = training_result.X_test
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)


def test_alias_defaults_to_model_name(registry, training_result):
    model_id = registry.save(training_result)
    _, meta = registry.load(model_id)
    assert meta["alias"] == training_result.model_name
