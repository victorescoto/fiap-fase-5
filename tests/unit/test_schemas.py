import pytest

from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringResponse,
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


class TestMonitoringResponse:
    def test_full_response(self) -> None:
        resp = MonitoringResponse(
            total_predictions=100,
            prediction_distribution={"baixo": 0.5, "medio": 0.3, "alto": 0.2},
            avg_confidence=0.85,
            drift_status={
                "is_drifted": False,
                "severity": "none",
                "max_difference": 0.05,
                "details": {},
            },
            recent_predictions=[
                {"prediction": "baixo", "probability": 0.9, "timestamp": "2026-01-01T00:00:00"}
            ],
        )
        assert resp.total_predictions == 100
        assert resp.prediction_distribution["baixo"] == 0.5
        assert resp.avg_confidence == 0.85
        assert resp.drift_status["is_drifted"] is False
        assert len(resp.recent_predictions) == 1

    def test_defaults(self) -> None:
        resp = MonitoringResponse(total_predictions=0)
        assert resp.prediction_distribution == {}
        assert resp.avg_confidence == 0.0
        assert resp.drift_status == {}
        assert resp.recent_predictions == []

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(Exception):
            MonitoringResponse()  # type: ignore[call-arg]


class TestBatchPredictRequest:
    def test_valid_batch(self) -> None:
        req = BatchPredictRequest(
            predictions=[
                PredictRequest(features={"a": 1}),
                PredictRequest(features={"b": 2}),
            ]
        )
        assert len(req.predictions) == 2

    def test_empty_list_raises(self) -> None:
        with pytest.raises(Exception):
            BatchPredictRequest(predictions=[])

    def test_missing_predictions_raises(self) -> None:
        with pytest.raises(Exception):
            BatchPredictRequest()  # type: ignore[call-arg]

    def test_invalid_item_type_raises(self) -> None:
        with pytest.raises(Exception):
            BatchPredictRequest(predictions=["not a PredictRequest"])  # type: ignore[list-item]


class TestBatchPredictResponse:
    def test_valid_response(self) -> None:
        resp = BatchPredictResponse(
            predictions=[
                PredictResponse(prediction="baixo", probability=0.9, model_version="1.0"),
                PredictResponse(prediction="medio", probability=0.7, model_version="1.0"),
            ]
        )
        assert len(resp.predictions) == 2
        assert resp.predictions[0].prediction == "baixo"

    def test_empty_predictions_list(self) -> None:
        resp = BatchPredictResponse(predictions=[])
        assert resp.predictions == []

    def test_missing_field_raises(self) -> None:
        with pytest.raises(Exception):
            BatchPredictResponse()  # type: ignore[call-arg]
