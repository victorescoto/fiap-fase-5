"""Prediction service layer.

Encapsulates the business logic for running predictions — building
the input ``DataFrame``, invoking the model, extracting probabilities
and logging the result.  This keeps :mod:`app.routes` thin (HTTP
concerns only).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.schemas import PredictResponse

logger = logging.getLogger(__name__)


def do_prediction(
    model: Any,
    features: dict[str, Any],
    metadata: dict[str, Any],
    prediction_logger: Any | None,
) -> PredictResponse:
    """Run a single prediction and optionally log it.

    Builds a single-row ``DataFrame`` so that sklearn ``Pipeline``
    objects with a ``ColumnTransformer`` preprocessor can match
    columns by name.

    Args:
        model: Trained model (must expose ``.predict()``).
        features: Raw feature dict sent by the client.
        metadata: Model metadata (used to obtain the version).
        prediction_logger: Optional :class:`PredictionLogger` instance.

    Returns:
        A fully populated :class:`PredictResponse`.

    Raises:
        Any exception propagated by the model — the caller is
        responsible for catching and returning an appropriate HTTP
        response.
    """
    input_df = pd.DataFrame([features])
    raw_prediction = model.predict(input_df)

    prediction = raw_prediction[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        probability = float(proba.max())

    version = metadata.get("version", "unknown")

    if prediction_logger is not None:
        prediction_logger.log_prediction(
            features=features,
            prediction=prediction,
            probability=probability,
            model_version=version,
        )

    return PredictResponse(
        prediction=prediction,
        probability=probability,
        model_version=version,
    )
