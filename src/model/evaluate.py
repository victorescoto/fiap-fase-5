"""
Model evaluation module for school delay prediction.

This module contains functions to:
- Evaluate models with multiple metrics
- Generate classification reports
- Calculate and display confusion matrix
- Calculate recall per class
"""

import logging
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Class order for consistency
CLASS_ORDER = ["baixo", "medio", "alto"]


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Calculates all evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict with all calculated metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0,
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0,
        ),
    }

    # Recall per class
    recalls = recall_score(
        y_true, y_pred, average=None, labels=CLASS_ORDER, zero_division=0,
    )
    for i, classe in enumerate(CLASS_ORDER):
        metrics[f"recall_{classe}"] = recalls[i]

    # Precision per class
    precisions = precision_score(
        y_true, y_pred, average=None, labels=CLASS_ORDER, zero_division=0,
    )
    for i, classe in enumerate(CLASS_ORDER):
        metrics[f"precision_{classe}"] = precisions[i]

    # F1 per class
    f1s = f1_score(
        y_true, y_pred, average=None, labels=CLASS_ORDER, zero_division=0,
    )
    for i, classe in enumerate(CLASS_ORDER):
        metrics[f"f1_{classe}"] = f1s[i]

    return metrics


def format_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> pd.DataFrame:
    """
    Formats the confusion matrix as DataFrame for better visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        DataFrame with formatted confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Real: {c}" for c in CLASS_ORDER],
        columns=[f"Pred: {c}" for c in CLASS_ORDER],
    )

    return cm_df


def print_evaluation_report(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    metrics: Dict[str, float],
) -> None:
    """
    Prints complete evaluation report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metrics: Dict with calculated metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("📊 CLASSIFICATION REPORT")
    logger.info("=" * 60)

    # Classification report from sklearn
    logger.info("\n" + classification_report(y_true,
          y_pred, labels=CLASS_ORDER, digits=4, zero_division=0))

    # Confusion matrix
    logger.info("\n" + "-" * 60)
    logger.info("🔢 CONFUSION MATRIX")
    logger.info("-" * 60)
    cm_df = format_confusion_matrix(y_true, y_pred)
    logger.info(cm_df.to_string())

    # Main metrics
    logger.info("\n" + "-" * 60)
    logger.info("📈 MAIN METRICS")
    logger.info("-" * 60)
    logger.info(f"  F1 Macro:     {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted:  {metrics['f1_weighted']:.4f}")
    logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")

    # Recall per class
    logger.info("\n" + "-" * 60)
    logger.info("🎯 RECALL PER CLASS")
    logger.info("-" * 60)
    for classe in CLASS_ORDER:
        recall = metrics[f"recall_{classe}"]
        logger.info(f"  {classe.capitalize():6}: {recall:.4f}")

    # Precision per class
    logger.info("\n" + "-" * 60)
    logger.info("🎯 PRECISION PER CLASS")
    logger.info("-" * 60)
    for classe in CLASS_ORDER:
        precision = metrics[f"precision_{classe}"]
        logger.info(f"  {classe.capitalize():6}: {precision:.4f}")

    # F1 per class
    logger.info("\n" + "-" * 60)
    logger.info("🎯 F1 PER CLASS")
    logger.info("-" * 60)
    for classe in CLASS_ORDER:
        f1 = metrics[f"f1_{classe}"]
        logger.info(f"  {classe.capitalize():6}: {f1:.4f}")


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluates a trained model on the test set.

    Calculates and prints (if verbose=True):
    - Complete classification report
    - Confusion matrix
    - F1 macro (main metric)
    - Recall per class

    Args:
        model: Trained pipeline (preprocessor + classifier)
        X_test: Test features
        y_test: Test target
        verbose: If True, prints reports

    Returns:
        Dict with all calculated metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)

    # Print report if verbose
    if verbose:
        print_evaluation_report(y_test, y_pred, metrics)

    return metrics


def get_predictions_with_probabilities(
    model: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns predictions with probabilities for each class.

    Useful for uncertainty analysis and threshold tuning.

    Args:
        model: Trained pipeline
        X: Features for prediction

    Returns:
        DataFrame with predictions and probabilities
    """
    predictions = model.predict(X)

    result = pd.DataFrame({"prediction": predictions})

    # Add probabilities if available
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X)
            for i, classe in enumerate(CLASS_ORDER):
                result[f"prob_{classe}"] = probas[:, i]
        except Exception:
            pass

    return result


def analyze_errors(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    feature_names: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Analyzes model prediction errors.

    Args:
        model: Trained pipeline
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names

    Returns:
        Dict with error analysis DataFrames
    """
    y_pred = model.predict(X_test)

    # Identify errors
    errors_mask = y_test != y_pred

    # DataFrame with error information
    error_analysis = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "is_error": errors_mask,
        }
    )

    # Matrix of error types
    error_types = pd.crosstab(
        error_analysis[error_analysis["is_error"]]["y_true"],
        error_analysis[error_analysis["is_error"]]["y_pred"],
        rownames=["Real"],
        colnames=["Predito"],
    )

    # Summary per class
    summary = []
    for classe in CLASS_ORDER:
        mask_classe = y_test == classe
        total = mask_classe.sum()
        corretos = ((y_test == classe) & (y_pred == classe)).sum()
        erros = ((y_test == classe) & (y_pred != classe)).sum()

        summary.append(
            {
                "classe": classe,
                "total": total,
                "corretos": corretos,
                "erros": erros,
                "taxa_acerto": corretos / total if total > 0 else 0,
                "taxa_erro": erros / total if total > 0 else 0,
            }
        )

    summary_df = pd.DataFrame(summary)

    return {
        "error_analysis": error_analysis,
        "error_types": error_types,
        "summary": summary_df,
    }


def compare_model_evaluations(
    evaluations: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Compares evaluations of multiple models in tabular format.

    Args:
        evaluations: Dict with model name and its metrics

    Returns:
        Comparative DataFrame
    """
    comparison = pd.DataFrame(evaluations).T

    # Sort by F1 macro
    comparison = comparison.sort_values("f1_macro", ascending=False)

    return comparison
