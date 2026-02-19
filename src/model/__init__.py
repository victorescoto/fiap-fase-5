"""
Módulo de Machine Learning para previsão de defasagem escolar.

Este módulo contém funções para:
- feature_engineering: Criação e transformação de features
- preprocessing: Pré-processamento de dados
- train: Treinamento e seleção de modelos
- evaluate: Avaliação de modelos
- predict: Inferência e predições (para integração com API)
"""

from .feature_engineering import build_features, create_target_column
from .preprocessing import build_preprocessor
from .train import train_model
from .evaluate import evaluate_model
from .predict import ModelPredictor, get_predictor, predict

__all__ = [
    # Feature Engineering
    "build_features",
    "create_target_column",
    # Preprocessing
    "build_preprocessor",
    # Training
    "train_model",
    # Evaluation
    "evaluate_model",
    # Prediction (API integration)
    "ModelPredictor",
    "get_predictor",
    "predict",
]
