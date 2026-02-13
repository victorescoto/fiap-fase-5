import pytest

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


class TestPredictRequest:
    def test_valid_features(self) -> None:
        req = PredictRequest(features={"a": 1, "b": 2.5, "c": "x"})
        assert req.features == {"a": 1, "b": 2.5, "c": "x"}

    def test_empty_features_dict(self) -> None:
        req = PredictRequest(features={})
        assert req.features == {}

    def test_missing_features_raises(self) -> None:
        with pytest.raises(Exception):
            PredictRequest()  # type: ignore[call-arg]

    def test_invalid_features_type_raises(self) -> None:
        with pytest.raises(Exception):
            PredictRequest(features="not a dict")  # type: ignore[arg-type]


class TestPredictResponse:
    def test_with_probability(self) -> None:
        resp = PredictResponse(prediction=1, probability=0.85, model_version="1.0.0")
        assert resp.prediction == 1
        assert resp.probability == 0.85
        assert resp.model_version == "1.0.0"

    def test_without_probability(self) -> None:
        resp = PredictResponse(prediction="low_risk", model_version="1.0.0")
        assert resp.probability is None

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(Exception):
            PredictResponse()  # type: ignore[call-arg]


class TestHealthResponse:
    def test_healthy(self) -> None:
        resp = HealthResponse(status="healthy", model_loaded=True)
        assert resp.status == "healthy"
        assert resp.model_loaded is True

    def test_unhealthy(self) -> None:
        resp = HealthResponse(status="healthy", model_loaded=False)
        assert resp.model_loaded is False


class TestModelInfoResponse:
    def test_full_info(self) -> None:
        resp = ModelInfoResponse(
            version="2.0.0",
            metrics={"accuracy": 0.9},
            features=["f1", "f2"],
        )
        assert resp.version == "2.0.0"
        assert resp.metrics == {"accuracy": 0.9}
        assert resp.features == ["f1", "f2"]

    def test_defaults(self) -> None:
        resp = ModelInfoResponse(version="1.0.0")
        assert resp.metrics == {}
        assert resp.features == []
