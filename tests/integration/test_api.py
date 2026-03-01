"""Integration tests for the FastAPI application endpoints.

Tests exercise each endpoint through the full HTTP stack using
``TestClient``, validating status codes, response shapes, and
error handling.
"""

from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_model_not_loaded(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is False

    def test_health_model_loaded(self, client_with_model: TestClient) -> None:
        response = client_with_model.get("/health")
        data = response.json()
        assert data["model_loaded"] is True


# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_valid_input(self, client_with_model: TestClient) -> None:
        response = client_with_model.post(
            "/api/v1/predict",
            json={"features": {"feature_1": 0.5, "feature_2": 1.0}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert data["model_version"] == "1.0.0-test"

    def test_predict_returns_probability(self, client_with_model: TestClient) -> None:
        response = client_with_model.post(
            "/api/v1/predict",
            json={"features": {"feature_1": 0.5, "feature_2": 1.0}},
        )
        data = response.json()
        assert data["probability"] is not None
        assert isinstance(data["probability"], float)

    def test_predict_invalid_input_missing_features(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_input_wrong_type(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict", json={"features": "not a dict"})
        assert response.status_code == 422

    def test_predict_model_not_loaded(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"features": {"feature_1": 1.0}},
        )
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_model_error_returns_500(
        self, client_with_model: TestClient, mock_model: MagicMock
    ) -> None:
        mock_model.predict.side_effect = ValueError("bad input shape")
        response = client_with_model.post(
            "/api/v1/predict",
            json={"features": {"feature_1": 1.0, "feature_2": 2.0}},
        )
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]

    def test_predict_missing_required_features_returns_422(
        self, client_with_model: TestClient
    ) -> None:
        """Sending only one of the two expected features should be rejected."""
        response = client_with_model.post(
            "/api/v1/predict",
            json={"features": {"feature_1": 0.5}},
        )
        assert response.status_code == 422
        data = response.json()
        assert "missing" in data["detail"].lower() or "missing" in str(data).lower()

    def test_predict_empty_features_rejected(
        self, client_with_model: TestClient
    ) -> None:
        """Empty features dict should be rejected when model expects features."""
        response = client_with_model.post(
            "/api/v1/predict",
            json={"features": {}},
        )
        assert response.status_code == 422

    def test_predict_extra_features_accepted_with_warning(
        self, client_with_model: TestClient
    ) -> None:
        """Extra features beyond expected should still succeed (logged as warning)."""
        response = client_with_model.post(
            "/api/v1/predict",
            json={
                "features": {
                    "feature_1": 0.5,
                    "feature_2": 1.0,
                    "UNKNOWN_EXTRA": 99,
                }
            },
        )
        # Should succeed — extras are ignored with a warning
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data


# ------------------------------------------------------------------
# Model Info
# ------------------------------------------------------------------


class TestModelInfoEndpoint:
    def test_model_info_with_metadata(self, client_with_model: TestClient) -> None:
        response = client_with_model.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.0.0-test"
        assert data["metrics"] == {"accuracy": 0.95}
        assert data["features"] == ["numeric__feature_1", "numeric__feature_2"]

    def test_model_info_without_metadata(self, client: TestClient) -> None:
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "unknown"


# ------------------------------------------------------------------
# Monitoring Stats
# ------------------------------------------------------------------


class TestMonitoringStatsEndpoint:
    def test_monitoring_stats_no_predictions(
        self, client_with_model: TestClient
    ) -> None:
        """Without any predictions, stats should be zeros."""
        response = client_with_model.get("/api/v1/monitoring/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] == 0
        assert data["prediction_distribution"] == {}
        assert data["avg_confidence"] == 0.0
        assert data["recent_predictions"] == []

    def test_monitoring_stats_after_predictions(
        self, client_with_model: TestClient
    ) -> None:
        """After making predictions, stats should reflect them."""
        # Make two predictions
        for _ in range(2):
            client_with_model.post(
                "/api/v1/predict",
                json={"features": {"feature_1": 0.5, "feature_2": 1.0}},
            )

        response = client_with_model.get("/api/v1/monitoring/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] == 2
        assert len(data["prediction_distribution"]) > 0
        assert data["avg_confidence"] > 0.0
        assert len(data["recent_predictions"]) == 2

    def test_monitoring_stats_drift_status_keys(
        self, client_with_model: TestClient
    ) -> None:
        """Drift status must contain expected keys."""
        # Make a prediction to populate stats
        client_with_model.post(
            "/api/v1/predict",
            json={"features": {"feature_1": 0.5, "feature_2": 1.0}},
        )
        response = client_with_model.get("/api/v1/monitoring/stats")
        data = response.json()
        drift = data["drift_status"]
        assert "is_drifted" in drift
        assert "severity" in drift
        assert "max_difference" in drift

    def test_monitoring_stats_recent_predictions_no_features(
        self, client_with_model: TestClient
    ) -> None:
        """Recent predictions in the stats payload should NOT include raw features."""
        client_with_model.post(
            "/api/v1/predict",
            json={"features": {"feature_1": 0.5, "feature_2": 1.0}},
        )
        response = client_with_model.get("/api/v1/monitoring/stats")
        data = response.json()
        for pred in data["recent_predictions"]:
            assert "features" not in pred

    def test_monitoring_stats_without_model(self, client: TestClient) -> None:
        """Without prediction_logger, should return safe defaults."""
        response = client.get("/api/v1/monitoring/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] == 0


# ------------------------------------------------------------------
# Batch Predict
# ------------------------------------------------------------------


class TestBatchPredictEndpoint:
    def test_batch_predict_valid(self, client_with_model: TestClient) -> None:
        response = client_with_model.post(
            "/api/v1/predict/batch",
            json={
                "predictions": [
                    {"features": {"feature_1": 0.5, "feature_2": 1.0}},
                    {"features": {"feature_1": 0.3, "feature_2": 0.8}},
                    {"features": {"feature_1": 0.7, "feature_2": 0.2}},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "model_version" in pred
            assert "probability" in pred

    def test_batch_predict_empty_list_rejected(
        self, client_with_model: TestClient
    ) -> None:
        response = client_with_model.post(
            "/api/v1/predict/batch",
            json={"predictions": []},
        )
        assert response.status_code == 422

    def test_batch_predict_model_not_loaded(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/predict/batch",
            json={
                "predictions": [
                    {"features": {"feature_1": 1.0}},
                ]
            },
        )
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_batch_predict_missing_features_rejected(
        self, client_with_model: TestClient
    ) -> None:
        """If any item in the batch has missing features, reject the batch."""
        response = client_with_model.post(
            "/api/v1/predict/batch",
            json={
                "predictions": [
                    {"features": {"feature_1": 0.5, "feature_2": 1.0}},
                    {"features": {"feature_1": 0.5}},  # missing feature_2
                ]
            },
        )
        assert response.status_code == 422

    def test_batch_predict_logs_to_monitoring(
        self, client_with_model: TestClient
    ) -> None:
        """Batch predictions should be logged to monitoring."""
        client_with_model.post(
            "/api/v1/predict/batch",
            json={
                "predictions": [
                    {"features": {"feature_1": 0.5, "feature_2": 1.0}},
                    {"features": {"feature_1": 0.3, "feature_2": 0.8}},
                ]
            },
        )
        response = client_with_model.get("/api/v1/monitoring/stats")
        data = response.json()
        assert data["total_predictions"] == 2
