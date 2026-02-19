"""
Módulo de avaliação de modelos para previsão de defasagem escolar.

Este módulo contém funções para:
- Avaliar modelos com múltiplas métricas
- Gerar relatórios de classificação
- Calcular e exibir matriz de confusão
- Calcular recall por classe
"""

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

# Ordem das classes para consistência
CLASS_ORDER = ["baixo", "medio", "alto"]


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Calcula todas as métricas de avaliação.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos

    Returns:
        Dict com todas as métricas calculadas
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=CLASS_ORDER),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", labels=CLASS_ORDER
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", labels=CLASS_ORDER
        ),
    }

    # Recall por classe
    recalls = recall_score(
        y_true, y_pred, average=None, labels=CLASS_ORDER
    )
    for i, classe in enumerate(CLASS_ORDER):
        metrics[f"recall_{classe}"] = recalls[i]

    # Precision por classe
    precisions = precision_score(
        y_true, y_pred, average=None, labels=CLASS_ORDER
    )
    for i, classe in enumerate(CLASS_ORDER):
        metrics[f"precision_{classe}"] = precisions[i]

    # F1 por classe
    f1s = f1_score(y_true, y_pred, average=None, labels=CLASS_ORDER)
    for i, classe in enumerate(CLASS_ORDER):
        metrics[f"f1_{classe}"] = f1s[i]

    return metrics


def format_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> pd.DataFrame:
    """
    Formata a matriz de confusão como DataFrame para melhor visualização.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos

    Returns:
        DataFrame com a matriz de confusão formatada
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
    Imprime relatório completo de avaliação.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        metrics: Dict com métricas calculadas
    """
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 60)

    # Classification report do sklearn
    print("\n" + classification_report(y_true,
          y_pred, labels=CLASS_ORDER, digits=4))

    # Matriz de confusão
    print("\n" + "-" * 60)
    print("🔢 MATRIZ DE CONFUSÃO")
    print("-" * 60)
    cm_df = format_confusion_matrix(y_true, y_pred)
    print(cm_df.to_string())

    # Métricas principais
    print("\n" + "-" * 60)
    print("📈 MÉTRICAS PRINCIPAIS")
    print("-" * 60)
    print(f"  F1 Macro:     {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted:  {metrics['f1_weighted']:.4f}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")

    # Recall por classe
    print("\n" + "-" * 60)
    print("🎯 RECALL POR CLASSE")
    print("-" * 60)
    for classe in CLASS_ORDER:
        recall = metrics[f"recall_{classe}"]
        print(f"  {classe.capitalize():6}: {recall:.4f}")

    # Precision por classe
    print("\n" + "-" * 60)
    print("🎯 PRECISION POR CLASSE")
    print("-" * 60)
    for classe in CLASS_ORDER:
        precision = metrics[f"precision_{classe}"]
        print(f"  {classe.capitalize():6}: {precision:.4f}")

    # F1 por classe
    print("\n" + "-" * 60)
    print("🎯 F1 POR CLASSE")
    print("-" * 60)
    for classe in CLASS_ORDER:
        f1 = metrics[f"f1_{classe}"]
        print(f"  {classe.capitalize():6}: {f1:.4f}")


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Avalia um modelo treinado no conjunto de teste.

    Calcula e imprime (se verbose=True):
    - Classification report completo
    - Matriz de confusão
    - F1 macro (métrica principal)
    - Recall por classe

    Args:
        model: Pipeline treinado (preprocessor + classifier)
        X_test: Features de teste
        y_test: Target de teste
        verbose: Se True, imprime relatórios

    Returns:
        Dict com todas as métricas calculadas
    """
    # Realiza predições
    y_pred = model.predict(X_test)

    # Calcula métricas
    metrics = calculate_metrics(y_test, y_pred)

    # Imprime relatório se verbose
    if verbose:
        print_evaluation_report(y_test, y_pred, metrics)

    return metrics


def get_predictions_with_probabilities(
    model: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retorna predições com probabilidades para cada classe.

    Útil para análise de incerteza e threshold tuning.

    Args:
        model: Pipeline treinado
        X: Features para predição

    Returns:
        DataFrame com predições e probabilidades
    """
    predictions = model.predict(X)

    result = pd.DataFrame({"prediction": predictions})

    # Adiciona probabilidades se disponíveis
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
    Analisa os erros de predição do modelo.

    Args:
        model: Pipeline treinado
        X_test: Features de teste
        y_test: Target de teste
        feature_names: Lista de nomes das features

    Returns:
        Dict com DataFrames de análise de erros
    """
    y_pred = model.predict(X_test)

    # Identifica erros
    errors_mask = y_test != y_pred

    # DataFrame com informações de erro
    error_analysis = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "is_error": errors_mask,
        }
    )

    # Matriz de tipos de erro
    error_types = pd.crosstab(
        error_analysis[error_analysis["is_error"]]["y_true"],
        error_analysis[error_analysis["is_error"]]["y_pred"],
        rownames=["Real"],
        colnames=["Predito"],
    )

    # Resumo por classe
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
    Compara avaliações de múltiplos modelos em formato tabular.

    Args:
        evaluations: Dict com nome do modelo e suas métricas

    Returns:
        DataFrame comparativo
    """
    comparison = pd.DataFrame(evaluations).T

    # Ordena por F1 macro
    comparison = comparison.sort_values("f1_macro", ascending=False)

    return comparison
