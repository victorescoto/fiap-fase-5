import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringResponse,
    PredictRequest,
    PredictResponse,
)
from app.validation import validate_features

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _validate_request_features(
    features: dict[str, Any],
    metadata: dict[str, Any],
) -> JSONResponse | None:
    """Return a 422 ``JSONResponse`` when *features* are invalid, else ``None``.

    Validates ``features`` keys against the model's expected input
    feature list stored in *metadata* (``input_features`` key — the raw
    column names the pipeline preprocessor expects).  Falls back to
    ``features`` (post-preprocessing names) when ``input_features`` is
    absent, stripping ``numeric__`` / ``categorical__`` prefixes.

    Missing features cause a hard 422; extra features are tolerated
    (with a log warning).
    """
    expected = metadata.get("input_features") or metadata.get("features", [])
    if not expected:
        return None  # no metadata to validate against

    missing, extra = validate_features(features, expected)

    if extra:
        logger.warning("Extra features ignored: %s", extra)

    if missing:
        return JSONResponse(
            status_code=422,
            content={
                "detail": f"Missing required features: {missing}",
                "missing": missing,
                "expected": [
                    f.split("__", 1)[-1] if "__" in f else f for f in expected
                ],
            },
        )
    return None


def _do_prediction(
    model: Any,
    features: dict[str, Any],
    metadata: dict[str, Any],
    prediction_logger: Any | None,
) -> PredictResponse:
    """Run a single prediction and log it.

    Builds a single-row ``DataFrame`` so that sklearn ``Pipeline``
    objects with a ``ColumnTransformer`` preprocessor can match
    columns by name.

    Raises on model errors — the caller is responsible for catching.
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


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request) -> HealthResponse:
    """Check API health and model loading status."""
    model_loaded = getattr(request.app.state, "model", None) is not None
    return HealthResponse(status="healthy", model_loaded=model_loaded)


@router.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    responses={
        422: {"description": "Missing or invalid features"},
        503: {"description": "Model not loaded"},
    },
)
async def predict(
    request: Request, body: PredictRequest
) -> PredictResponse | JSONResponse:
    """Run prediction using the loaded ML model.

    Returns 503 if the model is not available, 422 if required
    features are missing.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        logger.error("Prediction requested but model is not loaded")
        return JSONResponse(
            status_code=503,
            content={"detail": "Model not loaded. Service unavailable."},
        )

    metadata = getattr(request.app.state, "metadata", {})

    # Validate features against model metadata
    error_response = _validate_request_features(body.features, metadata)
    if error_response is not None:
        return error_response

    try:
        start = time.perf_counter()
        prediction_logger = getattr(request.app.state, "prediction_logger", None)
        result = _do_prediction(model, body.features, metadata, prediction_logger)
        elapsed = time.perf_counter() - start
        logger.info(
            "Prediction completed in %.4fs | result=%s", elapsed, result.prediction
        )
        return result
    except Exception:
        logger.exception("Error during prediction")
        return JSONResponse(
            status_code=500,
            content={"detail": "Prediction failed. Check input features."},
        )


@router.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    responses={
        422: {"description": "Missing or invalid features in one or more items"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_batch(
    request: Request, body: BatchPredictRequest
) -> BatchPredictResponse | JSONResponse:
    """Run predictions for a batch of inputs.

    Returns 503 if the model is not available, 422 if any item has
    missing required features.  The entire batch is rejected on
    validation failure.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        logger.error("Batch prediction requested but model is not loaded")
        return JSONResponse(
            status_code=503,
            content={"detail": "Model not loaded. Service unavailable."},
        )

    metadata = getattr(request.app.state, "metadata", {})

    # Validate features for every item before running any prediction
    for idx, item in enumerate(body.predictions):
        error_response = _validate_request_features(item.features, metadata)
        if error_response is not None:
            return error_response

    try:
        start = time.perf_counter()
        prediction_logger = getattr(request.app.state, "prediction_logger", None)
        results: list[PredictResponse] = []
        for item in body.predictions:
            result = _do_prediction(model, item.features, metadata, prediction_logger)
            results.append(result)
        elapsed = time.perf_counter() - start
        logger.info(
            "Batch prediction (%d items) completed in %.4fs",
            len(results),
            elapsed,
        )
        return BatchPredictResponse(predictions=results)
    except Exception:
        logger.exception("Error during batch prediction")
        return JSONResponse(
            status_code=500,
            content={"detail": "Batch prediction failed. Check input features."},
        )


@router.get(
    "/api/v1/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
async def model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the loaded model."""
    metadata = getattr(request.app.state, "metadata", {})
    return ModelInfoResponse(
        version=metadata.get("version", "unknown"),
        metrics=metadata.get("metrics", {}),
        features=metadata.get("features", []),
    )


@router.get(
    "/api/v1/monitoring/stats",
    response_model=MonitoringResponse,
    tags=["Monitoring"],
)
async def monitoring_stats(request: Request) -> MonitoringResponse:
    """Return prediction statistics and drift detection results."""
    prediction_logger = getattr(request.app.state, "prediction_logger", None)
    if prediction_logger is None:
        return MonitoringResponse(
            total_predictions=0,
            prediction_distribution={},
            avg_confidence=0.0,
            drift_status={
                "is_drifted": False,
                "severity": "none",
                "max_difference": 0.0,
                "details": {},
                "message": "Monitoring not initialized",
            },
            recent_predictions=[],
        )
    stats = prediction_logger.get_statistics()
    return MonitoringResponse(**stats)
