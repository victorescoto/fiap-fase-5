"""
Model training module for school delay prediction.

This module contains functions to:
- Train LogisticRegression model for multiclass classification
- Perform stratified cross-validation
- Evaluate and save the trained model
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    recall_score,
    make_scorer,
)

from .feature_engineering import build_features
from .preprocessing import build_preprocessor
from .evaluate import evaluate_model

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5  # Number of folds for cross-validation
TARGET_COLUMN = "nivel_defasagem"
CLASS_ORDER = ["baixo", "medio", "alto"]

# Directory to save models
MODEL_DIR = Path(__file__).parent.parent.parent / "app" / "model"


def get_model() -> LogisticRegression:
    """
    Returns the configured LogisticRegression model.

    Configuration:
    - solver='lbfgs': Optimization algorithm
    - max_iter=1000: Maximum iterations for convergence
    - class_weight='balanced': Adjusts weights for unbalanced classes
    - random_state=42: Reproducibility

    Returns:
        Configured LogisticRegression instance
    """
    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def create_custom_scorers() -> Dict[str, Any]:
    """
    Creates scorers for model evaluation.

    Scorers included:
    - f1_macro: F1 score macro-averaged (main metric)
    - recall_alto: Recall specific to the "alto" class

    Returns:
        Dict with scorers for cross_validate
    """

    def recall_alto_scorer(y_true, y_pred):
        """Calculates recall specific to 'alto' class."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        recalls = recall_score(
            y_true, y_pred, average=None, labels=CLASS_ORDER)
        idx_alto = CLASS_ORDER.index("alto")
        return recalls[idx_alto]

    scorers = {
        "f1_macro": "f1_macro",
        "recall_macro": "recall_macro",
        "accuracy": "accuracy",
        "recall_alto": make_scorer(recall_alto_scorer),
    }

    return scorers


def perform_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """
    Performs stratified cross-validation with LogisticRegression.

    Args:
        pipeline: Complete pipeline (preprocessor + model)
        X: Training features
        y: Training target

    Returns:
        Dict with cross-validation metrics
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    scorers = create_custom_scorers()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scorers,
            return_train_score=False,
            n_jobs=-1,
        )

    metrics = {
        "f1_macro_mean": cv_results["test_f1_macro"].mean(),
        "f1_macro_std": cv_results["test_f1_macro"].std(),
        "recall_macro_mean": cv_results["test_recall_macro"].mean(),
        "recall_macro_std": cv_results["test_recall_macro"].std(),
        "recall_alto_mean": cv_results["test_recall_alto"].mean(),
        "recall_alto_std": cv_results["test_recall_alto"].std(),
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
    }

    return metrics


def save_model(
    pipeline: Pipeline,
    metrics: Dict[str, float],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Saves the trained model using joblib.

    Args:
        pipeline: Trained pipeline
        metrics: Dict with evaluation metrics
        output_dir: Output directory (default: MODEL_DIR)

    Returns:
        Path to the saved file
    """
    if output_dir is None:
        output_dir = MODEL_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the model
    model_path = output_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {
        "model_name": "LogisticRegression",
        "metrics": metrics,
        "class_order": CLASS_ORDER,
        "random_state": RANDOM_STATE,
    }
    metadata_path = output_dir / "model_metadata.joblib"
    joblib.dump(metadata, metadata_path)

    print(f"\n✅ Model saved at: {model_path}")
    print(f"✅ Metadata saved at: {metadata_path}")

    return model_path


def train_model(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Main LogisticRegression model training function.

    Executes the complete pipeline:
    1. Feature engineering
    2. Stratified train/test split
    3. Preprocessor construction
    4. Cross-validation
    5. Final training
    6. Evaluation on test set
    7. Model saving

    Args:
        df: DataFrame with raw data
        save: If True, saves the trained model
        output_dir: Directory to save the model

    Returns:
        Tuple containing:
        - Trained model pipeline
        - Dict with results and metrics
    """
    print("\n" + "=" * 60)
    print("🚀 STARTING MODEL TRAINING")
    print("   Model: LogisticRegression")
    print("=" * 60)

    # 1. Feature Engineering
    print("\n📐 Executing Feature Engineering...")
    X, y = build_features(df, include_target=True)

    if y is None:
        raise ValueError("Target was not created. Check the 'Defas' column")

    # Remove rows with missing target
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    print(f"  Total samples: {len(X)}")
    print(f"  Total features: {X.shape[1]}")
    print(f"\n  Target distribution:")
    for classe in CLASS_ORDER:
        count = (y == classe).sum()
        pct = count / len(y) * 100
        print(f"    {classe}: {count} ({pct:.1f}%)")

    # 2. Stratified split
    print("\n📊 Performing stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # 3. Preprocessor construction
    print("\n🔧 Building preprocessor...")
    preprocessor = build_preprocessor(X_train)

    # 4. Pipeline creation
    print("\n🔬 Creating Pipeline (Preprocessor + LogisticRegression)...")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", get_model()),
        ]
    )

    # 5. Cross-validation
    print("\n📈 Executing Cross-Validation (5 folds)...")
    cv_metrics = perform_cross_validation(pipeline, X_train, y_train)

    print(
        f"  F1 Macro:     {cv_metrics['f1_macro_mean']:.4f} ± {cv_metrics['f1_macro_std']:.4f}")
    print(
        f"  Recall Macro: {cv_metrics['recall_macro_mean']:.4f} ± {cv_metrics['recall_macro_std']:.4f}")
    print(
        f"  Recall Alto:  {cv_metrics['recall_alto_mean']:.4f} ± {cv_metrics['recall_alto_std']:.4f}")
    print(
        f"  Accuracy:     {cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}")

    # 6. Final training with all training data
    print("\n🏋️ Training final model...")
    pipeline.fit(X_train, y_train)

    # 7. Final evaluation on test set
    print("\n" + "=" * 60)
    print("📋 FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    test_metrics = evaluate_model(pipeline, X_test, y_test)

    # 8. Model saving
    if save:
        save_model(pipeline, test_metrics, output_dir)

    # Prepare results
    results = {
        "model_name": "LogisticRegression",
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": X.shape[1],
    }

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return pipeline, results


def load_model(model_path: Optional[Path] = None) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Loads a saved model.

    Args:
        model_path: Path to the model file (default: MODEL_DIR/model.joblib)

    Returns:
        Tuple containing:
        - Loaded model pipeline
        - Dict with metadata
    """
    if model_path is None:
        model_path = MODEL_DIR / "model.joblib"

    model_path = Path(model_path)
    metadata_path = model_path.parent / "model_metadata.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipeline = joblib.load(model_path)
    metadata = {}

    if metadata_path.exists():
        metadata = joblib.load(metadata_path)

    return pipeline, metadata
