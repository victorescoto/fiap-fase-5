"""
Módulo de Machine Learning para previsão de defasagem escolar.

Este módulo contém funções para:
- feature_engineering: Criação e transformação de features
- preprocessing: Pré-processamento de dados
- train: Treinamento e seleção de modelos
- evaluate: Avaliação de modelos
"""

from .feature_engineering import build_features, create_target_column
from .preprocessing import build_preprocessor
from .train import train_model
from .evaluate import evaluate_model

__all__ = [
    "build_features",
    "create_target_column",
    "build_preprocessor",
    "train_model",
    "evaluate_model",
]
