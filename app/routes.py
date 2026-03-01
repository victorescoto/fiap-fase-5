import logging
import time

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
from app.services import do_prediction
from app.validation import MissingFeaturesError, validate_request_features

logger = logging.getLogger(__name__)

router = APIRouter()


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

    try:
        validate_request_features(body.features, metadata)
    except MissingFeaturesError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc),
                "missing": exc.missing,
                "expected": exc.expected,
            },
        )

    try:
        start = time.perf_counter()
        prediction_logger = getattr(request.app.state, "prediction_logger", None)
        result = do_prediction(model, body.features, metadata, prediction_logger)
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

    # Validate every item before running any prediction
    for item in body.predictions:
        try:
            validate_request_features(item.features, metadata)
        except MissingFeaturesError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": str(exc),
                    "missing": exc.missing,
                    "expected": exc.expected,
                },
            )

    try:
        start = time.perf_counter()
        prediction_logger = getattr(request.app.state, "prediction_logger", None)
        results: list[PredictResponse] = []
        for item in body.predictions:
            result = do_prediction(model, item.features, metadata, prediction_logger)
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
