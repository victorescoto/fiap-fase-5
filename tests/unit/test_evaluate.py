"""Tests for src.model.evaluate module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.model.evaluate import (
    CLASS_ORDER,
    calculate_metrics,
    evaluate_model,
    format_confusion_matrix,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def perfect_predictions():
    """y_true and y_pred with perfect accuracy."""
    y_true = pd.Series(["baixo", "medio", "alto", "baixo", "medio", "alto"])
    y_pred = np.array(["baixo", "medio", "alto", "baixo", "medio", "alto"])
    return y_true, y_pred


@pytest.fixture
def imperfect_predictions():
    """y_true and y_pred with some errors."""
    y_true = pd.Series(["baixo", "medio", "alto", "baixo", "medio", "alto"])
    y_pred = np.array(["baixo", "medio", "medio", "baixo", "baixo", "alto"])
    return y_true, y_pred


# ------------------------------------------------------------------
# calculate_metrics
# ------------------------------------------------------------------


class TestCalculateMetrics:
    def test_perfect_returns_ones(self, perfect_predictions) -> None:
        y_true, y_pred = perfect_predictions
        m = calculate_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0
        assert m["recall_macro"] == 1.0
        assert m["precision_macro"] == 1.0

    def test_all_expected_keys(self, perfect_predictions) -> None:
        y_true, y_pred = perfect_predictions
        m = calculate_metrics(y_true, y_pred)
        expected_keys = [
            "accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro",
        ]
        for k in expected_keys:
            assert k in m
        for cls in CLASS_ORDER:
            assert f"recall_{cls}" in m
            assert f"precision_{cls}" in m
            assert f"f1_{cls}" in m

    def test_imperfect_metrics_below_one(self, imperfect_predictions) -> None:
        y_true, y_pred = imperfect_predictions
        m = calculate_metrics(y_true, y_pred)
        assert m["accuracy"] < 1.0
        assert m["f1_macro"] < 1.0


# ------------------------------------------------------------------
# format_confusion_matrix
# ------------------------------------------------------------------


class TestFormatConfusionMatrix:
    def test_returns_dataframe(self, perfect_predictions) -> None:
        y_true, y_pred = perfect_predictions
        cm = format_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, pd.DataFrame)
        assert cm.shape == (3, 3)

    def test_perfect_is_diagonal(self, perfect_predictions) -> None:
        y_true, y_pred = perfect_predictions
        cm = format_confusion_matrix(y_true, y_pred)
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert cm.iloc[i, j] > 0
                else:
                    assert cm.iloc[i, j] == 0


# ------------------------------------------------------------------
# evaluate_model
# ------------------------------------------------------------------


class TestEvaluateModel:
    def test_returns_metrics_dict(self) -> None:
        model = MagicMock()
        model.predict.return_value = np.array(["baixo", "medio", "alto"])
        X_test = pd.DataFrame({"a": [1, 2, 3]})
        y_test = pd.Series(["baixo", "medio", "alto"])

        metrics = evaluate_model(model, X_test, y_test, verbose=False)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        model.predict.assert_called_once()

    def test_verbose_does_not_crash(self) -> None:
        model = MagicMock()
        model.predict.return_value = np.array(["baixo", "medio"])
        X = pd.DataFrame({"a": [1, 2]})
        y = pd.Series(["baixo", "medio"])
        # Should not raise
        evaluate_model(model, X, y, verbose=True)
