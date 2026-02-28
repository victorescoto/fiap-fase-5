"""
Machine Learning module for school delay prediction.

This module contains functions for:
- feature_engineering: Feature creation and transformation
- preprocessing: Data preprocessing
- train: Model training and selection
- evaluate: Model evaluation
- predict: Inference and predictions (for API integration)
"""

from .feature_engineering import build_features, create_target_column
from .preprocessing import build_preprocessor
from .train import train_model
from .evaluate import evaluate_model
# from .predict import ModelPredictor, get_predictor, predict  # TODO: Create predict module

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
    # TODO: Add prediction exports when module is created
    # "ModelPredictor",
    # "get_predictor",
    # "predict",
]
