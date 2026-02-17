import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient]:
    """TestClient with no model loaded (degraded mode)."""
    with (
        patch("app.main.load_model", return_value=None),
        patch(
            "app.main.load_metadata",
            return_value={"version": "unknown", "metrics": {}, "features": []},
        ),
        TestClient(app) as c,
    ):
        yield c


@pytest.fixture
def mock_model() -> MagicMock:
    """A mock sklearn-like model with predict and predict_proba."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


@pytest.fixture
def client_with_model(mock_model: MagicMock) -> Generator[TestClient]:
    """TestClient with a mock model loaded."""
    metadata = {
        "version": "1.0.0-test",
        "metrics": {"accuracy": 0.95},
        "features": ["feature_1", "feature_2"],
    }
    with (
        patch("app.main.load_model", return_value=mock_model),
        patch("app.main.load_metadata", return_value=metadata),
        TestClient(app) as c,
    ):
        yield c


class _DummyModel:
    """A simple picklable dummy model for serialization tests."""

    def predict(self, x):  # noqa: ANN001, ANN201
        return np.array([0] * len(x))

    def predict_proba(self, x):  # noqa: ANN001, ANN201
        return np.array([[0.5, 0.5]] * len(x))


@pytest.fixture
def tmp_model_path() -> Generator[Path]:
    """Create a temporary model file serialized with joblib."""
    model = _DummyModel()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        joblib.dump(model, f.name)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def tmp_metadata_path() -> Generator[Path]:
    """Create a temporary metadata JSON file."""
    metadata = {
        "version": "1.0.0",
        "metrics": {"accuracy": 0.9},
        "features": ["a", "b"],
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(metadata, f)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)
