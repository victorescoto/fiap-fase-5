"""
Prediction monitoring module for drift detection.

This module provides:
- PredictionLogger: Thread-safe in-memory prediction store
- Drift detection via prediction distribution comparison
- Statistics aggregation for the monitoring dashboard
"""

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE = 1000
DRIFT_THRESHOLD = 0.15  # Max acceptable difference per class distribution


class PredictionLogger:
    """Thread-safe in-memory store for predictions and drift monitoring.

    Keeps the last ``max_size`` predictions in a bounded deque and
    provides aggregated statistics for the monitoring dashboard.

    Args:
        baseline_stats: Dict with ``prediction_distribution`` and
            ``avg_confidence`` from the training metadata.
        max_size: Maximum number of predictions to retain.
    """

    def __init__(
        self,
        baseline_stats: dict[str, Any] | None = None,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> None:
        self._lock = threading.Lock()
        self._predictions: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._max_size = max_size
        self.baseline_stats = baseline_stats or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        features: dict[str, Any],
        prediction: Any,
        probability: float | None = None,
        model_version: str = "unknown",
    ) -> None:
        """Record a single prediction event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction,
            "probability": probability,
            "model_version": model_version,
            "features": features,
        }
        with self._lock:
            self._predictions.append(entry)
        logger.debug("Logged prediction: %s (prob=%.4f)", prediction, probability or 0)

    @property
    def count(self) -> int:
        """Number of predictions currently stored."""
        with self._lock:
            return len(self._predictions)

    def get_recent_predictions(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the *n* most recent prediction records."""
        with self._lock:
            items = list(self._predictions)
        return items[-n:]

    def get_statistics(self) -> dict[str, Any]:
        """Aggregate statistics over stored predictions.

        Returns a dict with:
        - total_predictions
        - prediction_distribution  (label → proportion)
        - avg_confidence
        - drift_status  (from ``check_drift()``)
        - recent_predictions  (last 10)
        """
        with self._lock:
            items = list(self._predictions)

        total = len(items)
        if total == 0:
            return {
                "total_predictions": 0,
                "prediction_distribution": {},
                "avg_confidence": 0.0,
                "drift_status": self._no_data_drift_status(),
                "recent_predictions": [],
            }

        # Prediction distribution
        labels = [p["prediction"] for p in items]
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {str(label): int(c) / total for label, c in zip(unique, counts)}

        # Avg confidence
        probs = [p["probability"] for p in items if p["probability"] is not None]
        avg_conf = float(np.mean(probs)) if probs else 0.0

        drift_status = self.check_drift(distribution, avg_conf)

        # Strip features from recent entries for lighter payload
        recent = [
            {k: v for k, v in p.items() if k != "features"}
            for p in items[-10:]
        ]

        return {
            "total_predictions": total,
            "prediction_distribution": distribution,
            "avg_confidence": round(avg_conf, 4),
            "drift_status": drift_status,
            "recent_predictions": recent,
        }

    def check_drift(
        self,
        current_distribution: dict[str, float] | None = None,
        current_avg_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Compare current prediction distribution against baseline.

        Returns a dict with drift detection results:
        - is_drifted (bool)
        - severity ("none" | "warning" | "critical")
        - details per class
        """
        baseline_dist = self.baseline_stats.get("prediction_distribution", {})

        if not baseline_dist or current_distribution is None:
            return self._no_data_drift_status()

        all_labels = set(list(baseline_dist.keys()) + list(current_distribution.keys()))

        diffs: dict[str, float] = {}
        for label in all_labels:
            baseline_val = baseline_dist.get(label, 0.0)
            current_val = current_distribution.get(label, 0.0)
            diffs[label] = round(current_val - baseline_val, 4)

        max_diff = max(abs(d) for d in diffs.values()) if diffs else 0.0

        if max_diff >= DRIFT_THRESHOLD * 2:
            severity = "critical"
        elif max_diff >= DRIFT_THRESHOLD:
            severity = "warning"
        else:
            severity = "none"

        is_drifted = severity != "none"

        result = {
            "is_drifted": is_drifted,
            "severity": severity,
            "max_difference": round(max_diff, 4),
            "details": {
                label: {
                    "baseline": round(baseline_dist.get(label, 0.0), 4),
                    "current": round(current_distribution.get(label, 0.0), 4),
                    "difference": diffs[label],
                }
                for label in sorted(all_labels)
            },
        }

        if is_drifted:
            logger.warning("Drift detected: severity=%s diffs=%s", severity, diffs)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_data_drift_status() -> dict[str, Any]:
        return {
            "is_drifted": False,
            "severity": "none",
            "max_difference": 0.0,
            "details": {},
            "message": "Not enough data for drift detection",
        }
