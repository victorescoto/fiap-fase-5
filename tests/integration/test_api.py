from unittest.mock import MagicMock

from fastapi.testclient import TestClient


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
            json={"features": {"feature_1": 1.0}},
        )
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]


class TestModelInfoEndpoint:
    def test_model_info_with_metadata(self, client_with_model: TestClient) -> None:
        response = client_with_model.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.0.0-test"
        assert data["metrics"] == {"accuracy": 0.95}
        assert data["features"] == ["feature_1", "feature_2"]

    def test_model_info_without_metadata(self, client: TestClient) -> None:
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "unknown"
