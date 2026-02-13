import logging
import time

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request) -> HealthResponse:
    """Check API health and model loading status."""
    model_loaded = getattr(request.app.state, "model", None) is not None
    return HealthResponse(status="healthy", model_loaded=model_loaded)


@router.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    responses={503: {"description": "Model not loaded"}},
)
async def predict(
    request: Request, body: PredictRequest
) -> PredictResponse | JSONResponse:
    """Run prediction using the loaded ML model.

    Returns 503 if the model is not available.
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
        start = time.perf_counter()

        features = body.features
        feature_values = np.array([list(features.values())])
        raw_prediction = model.predict(feature_values)

        prediction = raw_prediction[0]
        # Convert numpy types to native Python types for JSON serialization
        if hasattr(prediction, "item"):
            prediction = prediction.item()

        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feature_values)
            probability = float(proba.max())

        elapsed = time.perf_counter() - start
        logger.info("Prediction completed in %.4fs | result=%s", elapsed, prediction)

        return PredictResponse(
            prediction=prediction,
            probability=probability,
            model_version=metadata.get("version", "unknown"),
        )
    except Exception:
        logger.exception("Error during prediction")
        return JSONResponse(
            status_code=500,
            content={"detail": "Prediction failed. Check input features."},
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
