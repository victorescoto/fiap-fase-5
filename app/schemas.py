from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request body for the prediction endpoint.

    Accepts a dictionary mapping feature names to values.
    Feature names should use the *raw* column names (without the
    ``numeric__`` / ``categorical__`` prefix added by the
    preprocessor).  The API strips these prefixes automatically
    when validating against the model metadata.
    """

    features: dict[str, Any] = Field(
        ...,
        description="Dictionary of student features for prediction",
        json_schema_extra={
            "example": {
                "Fase": 4,
                "Ano nasc": 2010,
                "Idade 22": 12,
                "Ano ingresso": 2020,
                "IAA": 7.5,
                "IEG": 8.0,
                "IPS": 6.5,
                "IDA": 7.0,
                "IPV": 6.0,
                "IAN": 5.5,
                "Gênero_Menino": 1,
                "Instituição de ensino_Escola Pública": 1,
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


class BatchPredictRequest(BaseModel):
    """Request body for the batch prediction endpoint."""

    predictions: list[PredictRequest] = Field(
        ...,
        description="List of prediction requests to process in batch",
        min_length=1,
    )


class BatchPredictResponse(BaseModel):
    """Response body for the batch prediction endpoint."""

    predictions: list[PredictResponse] = Field(
        ..., description="List of prediction results",
    )
