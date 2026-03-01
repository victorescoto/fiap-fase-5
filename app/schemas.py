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
                "G\u00eanero": "Menino",
                "Ano ingresso": 2020,
                "Institui\u00e7\u00e3o de ensino": "Escola P\u00fablica",
                "INDE 22": 7.8,
                "Cg": 7.0,
                "Cf": 7.5,
                "Ct": 8.0,
                "N\u00ba Av": 4,
                "IAA": 7.5,
                "IEG": 8.0,
                "IPS": 6.5,
                "IDA": 7.0,
                "Matem": 7.0,
                "Portug": 6.5,
                "Ingl\u00eas": 8.0,
                "IPV": 6.0,
                "IAN": 5.5,
                "Pedra 20_encoded": 2,
                "Pedra 21_encoded": 3,
                "Pedra 22_encoded": 3,
                "tempo_no_programa": 3,
                "idade_ingresso": 9,
                "pedra_evolucao_20_21": 1,
                "pedra_evolucao_21_22": 0,
                "pedra_evolucao_total": 1,
                "media_disciplinas": 7.17,
                "std_disciplinas": 0.76,
                "min_disciplina": 6.5,
                "max_disciplina": 8.0,
                "media_indicadores": 6.75,
                "std_indicadores": 0.83,
                "ratio_inde_indicadores": 1.16,
                "diff_iaa_ida": 0.5,
                "diff_ieg_ips": 1.5,
                "indicado_bin": 0,
                "atingiu_pv_bin": 1,
                "psicologia_requer_avaliacao": 0
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
