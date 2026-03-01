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


# ------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------

_BASE_METADATA: dict = {
    "version": "unknown",
    "metrics": {},
    "features": [],
}

_MODEL_METADATA: dict = {
    "version": "1.0.0-test",
    "metrics": {"accuracy": 0.95},
    "features": ["numeric__feature_1", "numeric__feature_2"],
    "baseline_stats": {
        "prediction_distribution": {
            "baixo": 0.60,
            "medio": 0.30,
            "alto": 0.10,
        },
        "avg_confidence": 0.85,
        "total_samples": 100,
    },
}


# ------------------------------------------------------------------
# Clients
# ------------------------------------------------------------------

@pytest.fixture
def client() -> Generator[TestClient]:
    """TestClient with no model loaded (degraded mode)."""
    with (
        patch("app.main.load_model", return_value=None),
        patch("app.main.load_metadata", return_value={**_BASE_METADATA}),
        TestClient(app) as c,
    ):
        yield c


@pytest.fixture
def mock_model() -> MagicMock:
    """A mock sklearn-like model with predict and predict_proba."""
    model = MagicMock()
    model.predict.return_value = np.array(["baixo"])
    model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])
    return model


@pytest.fixture
def client_with_model(mock_model: MagicMock) -> Generator[TestClient]:
    """TestClient with a mock model loaded and baseline stats for monitoring."""
    with (
        patch("app.main.load_model", return_value=mock_model),
        patch("app.main.load_metadata", return_value={**_MODEL_METADATA}),
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
