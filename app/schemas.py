from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for the prediction endpoint.

    Uses a flexible dict for features until the model's expected
    input schema is finalized. Once defined, replace `features`
    with explicit typed fields.
    """

    features: dict[str, Any] = Field(
        ...,
        description="Dictionary of student features for prediction",
        json_schema_extra={
            "example": {
                "feature_1": 0.5,
                "feature_2": "value",
                "feature_3": 10,
            }
        },
    )


class PredictResponse(BaseModel):
    """Response body for the prediction endpoint."""

    prediction: Any = Field(..., description="Model prediction value")
    probability: float | None = Field(
        None, description="Prediction probability (if classification model)"
    )
    model_version: str = Field(..., description="Version of the model used")


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(
        ..., description="Whether the ML model is currently loaded"
    )


class ModelInfoResponse(BaseModel):
    """Response body for the model info endpoint."""

    version: str = Field(..., description="Model version")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Model evaluation metrics"
    )
    features: list[str] = Field(
        default_factory=list, description="Expected feature names"
    )


class MonitoringResponse(BaseModel):
    """Response body for the monitoring stats endpoint."""

    total_predictions: int = Field(
        ..., description="Total number of predictions logged"
    )
    prediction_distribution: dict[str, Any] = Field(
        default_factory=dict,
        description="Current prediction label distribution (label → proportion)",
    )
    avg_confidence: float = Field(
        0.0, description="Average prediction confidence/probability"
    )
    drift_status: dict[str, Any] = Field(
        default_factory=dict,
        description="Drift detection results (is_drifted, severity, details)",
    )
    recent_predictions: list[dict[str, Any]] = Field(
        default_factory=list, description="Most recent predictions"
    )
