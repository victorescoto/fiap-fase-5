"""Tests for app.monitoring module."""

import pytest

from app.monitoring import PredictionLogger, DRIFT_THRESHOLD


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def baseline_stats() -> dict:
    return {
        "prediction_distribution": {
            "baixo": 0.30,
            "medio": 0.60,
            "alto": 0.10,
        },
        "avg_confidence": 0.85,
    }


@pytest.fixture
def logger_with_baseline(baseline_stats: dict) -> PredictionLogger:
    return PredictionLogger(baseline_stats=baseline_stats, max_size=50)


@pytest.fixture
def empty_logger() -> PredictionLogger:
    return PredictionLogger(max_size=10)


# ------------------------------------------------------------------
# log_prediction & count
# ------------------------------------------------------------------


class TestLogPrediction:
    def test_count_increases(self, empty_logger: PredictionLogger) -> None:
        assert empty_logger.count == 0
        empty_logger.log_prediction({"a": 1}, "baixo", 0.9)
        assert empty_logger.count == 1

    def test_bounded_by_max_size(self, empty_logger: PredictionLogger) -> None:
        for i in range(20):
            empty_logger.log_prediction({"a": i}, "baixo", 0.5)
        assert empty_logger.count == 10  # max_size=10

    def test_entry_fields(self, empty_logger: PredictionLogger) -> None:
        empty_logger.log_prediction({"x": 5}, "alto", 0.7, model_version="2.0")
        recent = empty_logger.get_recent_predictions(1)
        entry = recent[0]
        assert entry["prediction"] == "alto"
        assert entry["probability"] == 0.7
        assert entry["model_version"] == "2.0"
        assert "timestamp" in entry
        assert entry["features"] == {"x": 5}

    def test_timestamp_is_utc_aware(self, empty_logger: PredictionLogger) -> None:
        """Timestamps must use timezone-aware UTC (not deprecated utcnow())."""
        from datetime import datetime, timezone

        empty_logger.log_prediction({"x": 1}, "baixo", 0.9)
        recent = empty_logger.get_recent_predictions(1)
        ts = recent[0]["timestamp"]
        # Should end with +00:00 (timezone-aware) or contain timezone info
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None, (
            f"Timestamp '{ts}' is naive (no timezone). Use datetime.now(timezone.utc)."
        )


# ------------------------------------------------------------------
# get_recent_predictions
# ------------------------------------------------------------------


class TestGetRecentPredictions:
    def test_returns_last_n(self, empty_logger: PredictionLogger) -> None:
        for i in range(5):
            empty_logger.log_prediction({"i": i}, f"label_{i}", 0.5)
        recent = empty_logger.get_recent_predictions(3)
        assert len(recent) == 3
        assert recent[-1]["prediction"] == "label_4"

    def test_empty_returns_empty(self, empty_logger: PredictionLogger) -> None:
        assert empty_logger.get_recent_predictions() == []


# ------------------------------------------------------------------
# get_statistics
# ------------------------------------------------------------------


class TestGetStatistics:
    def test_empty_stats(self, empty_logger: PredictionLogger) -> None:
        stats = empty_logger.get_statistics()
        assert stats["total_predictions"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_distribution_calculation(self, logger_with_baseline: PredictionLogger) -> None:
        for _ in range(3):
            logger_with_baseline.log_prediction({}, "baixo", 0.9)
        for _ in range(7):
            logger_with_baseline.log_prediction({}, "medio", 0.8)

        stats = logger_with_baseline.get_statistics()
        assert stats["total_predictions"] == 10
        assert stats["prediction_distribution"]["baixo"] == pytest.approx(0.3)
        assert stats["prediction_distribution"]["medio"] == pytest.approx(0.7)

    def test_avg_confidence(self, logger_with_baseline: PredictionLogger) -> None:
        logger_with_baseline.log_prediction({}, "baixo", 0.8)
        logger_with_baseline.log_prediction({}, "medio", 0.6)
        stats = logger_with_baseline.get_statistics()
        assert stats["avg_confidence"] == pytest.approx(0.7)

    def test_stats_include_drift_status(self, logger_with_baseline: PredictionLogger) -> None:
        logger_with_baseline.log_prediction({}, "baixo", 0.9)
        stats = logger_with_baseline.get_statistics()
        assert "drift_status" in stats
        assert "is_drifted" in stats["drift_status"]

    def test_recent_predictions_no_features(self, logger_with_baseline: PredictionLogger) -> None:
        """Recent predictions in stats should not include feature data."""
        logger_with_baseline.log_prediction({"x": 1}, "baixo", 0.9)
        stats = logger_with_baseline.get_statistics()
        for p in stats["recent_predictions"]:
            assert "features" not in p


# ------------------------------------------------------------------
# check_drift
# ------------------------------------------------------------------


class TestCheckDrift:
    def test_no_drift_when_close(self, logger_with_baseline: PredictionLogger) -> None:
        # Distribution very close to baseline
        current = {"baixo": 0.31, "medio": 0.59, "alto": 0.10}
        result = logger_with_baseline.check_drift(current)
        assert result["is_drifted"] is False
        assert result["severity"] == "none"

    def test_warning_drift(self, logger_with_baseline: PredictionLogger) -> None:
        # One class is off by threshold amount
        current = {"baixo": 0.50, "medio": 0.45, "alto": 0.05}
        result = logger_with_baseline.check_drift(current)
        assert result["is_drifted"] is True
        assert result["severity"] == "warning"

    def test_critical_drift(self, logger_with_baseline: PredictionLogger) -> None:
        # Massive distribution shift
        current = {"baixo": 0.0, "medio": 0.0, "alto": 1.0}
        result = logger_with_baseline.check_drift(current)
        assert result["is_drifted"] is True
        assert result["severity"] == "critical"

    def test_no_baseline_returns_safe(self, empty_logger: PredictionLogger) -> None:
        result = empty_logger.check_drift({"baixo": 1.0})
        assert result["is_drifted"] is False
        assert "message" in result

    def test_details_per_class(self, logger_with_baseline: PredictionLogger) -> None:
        current = {"baixo": 0.40, "medio": 0.50, "alto": 0.10}
        result = logger_with_baseline.check_drift(current)
        assert "alto" in result["details"]
        assert "baseline" in result["details"]["alto"]
        assert "current" in result["details"]["alto"]
        assert "difference" in result["details"]["alto"]
