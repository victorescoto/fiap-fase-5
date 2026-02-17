"""
Módulo de treinamento de modelos para previsão de defasagem escolar.

Este módulo contém funções para:
- Treinar modelo LogisticRegression para classificação multiclasse
- Realizar validação cruzada estratificada
- Avaliar e salvar o modelo treinado
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

# Configurações
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5  # Número de folds para cross-validation
TARGET_COLUMN = "nivel_defasagem"
CLASS_ORDER = ["baixo", "medio", "alto"]

# Diretório para salvar modelos
MODEL_DIR = Path(__file__).parent.parent.parent / "app" / "model"


def get_model() -> LogisticRegression:
    """
    Retorna o modelo LogisticRegression configurado.

    Configurações:
    - solver='lbfgs': Algoritmo de otimização
    - max_iter=1000: Iterações máximas para convergência
    - class_weight='balanced': Ajusta pesos para classes desbalanceadas
    - random_state=42: Reprodutibilidade

    Returns:
        Instância do LogisticRegression configurada
    """
    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def create_custom_scorers() -> Dict[str, Any]:
    """
    Cria scorers para avaliação do modelo.

    Scorers incluídos:
    - f1_macro: F1 score macro-averaged (métrica principal)
    - recall_alto: Recall específico para a classe "alto"

    Returns:
        Dict com scorers para cross_validate
    """

    def recall_alto_scorer(y_true, y_pred):
        """Calcula recall específico para a classe 'alto'."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        recalls = recall_score(y_true, y_pred, average=None, labels=CLASS_ORDER)
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
    Realiza validação cruzada estratificada com LogisticRegression.

    Args:
        pipeline: Pipeline completo (preprocessor + modelo)
        X: Features de treino
        y: Target de treino

    Returns:
        Dict com métricas da validação cruzada
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
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
    Salva o modelo treinado usando joblib.

    Args:
        pipeline: Pipeline treinado
        metrics: Dict com métricas de avaliação
        output_dir: Diretório de saída (default: MODEL_DIR)

    Returns:
        Path para o arquivo salvo
    """
    if output_dir is None:
        output_dir = MODEL_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva o modelo
    model_path = output_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Salva metadados
    metadata = {
        "model_name": "LogisticRegression",
        "metrics": metrics,
        "class_order": CLASS_ORDER,
        "random_state": RANDOM_STATE,
    }
    metadata_path = output_dir / "model_metadata.joblib"
    joblib.dump(metadata, metadata_path)

    print(f"\n✅ Modelo salvo em: {model_path}")
    print(f"✅ Metadados salvos em: {metadata_path}")

    return model_path


def train_model(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Função principal de treinamento do modelo LogisticRegression.

    Executa o pipeline completo:
    1. Feature engineering
    2. Split estratificado treino/teste
    3. Construção do preprocessador
    4. Validação cruzada
    5. Treinamento final
    6. Avaliação no conjunto de teste
    7. Salvamento do modelo

    Args:
        df: DataFrame com dados brutos
        save: Se True, salva o modelo treinado
        output_dir: Diretório para salvar o modelo

    Returns:
        Tuple contendo:
        - Pipeline do modelo treinado
        - Dict com resultados e métricas
    """
    print("\n" + "=" * 60)
    print("🚀 INICIANDO TREINAMENTO DO MODELO")
    print("   Modelo: LogisticRegression")
    print("=" * 60)

    # 1. Feature Engineering
    print("\n📐 Executando Feature Engineering...")
    X, y = build_features(df, include_target=True)

    if y is None:
        raise ValueError("Target não foi criado. Verifique a coluna 'Defas'")

    # Remove linhas com target missing
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    print(f"  Total de amostras: {len(X)}")
    print(f"  Total de features: {X.shape[1]}")
    print(f"\n  Distribuição do target:")
    for classe in CLASS_ORDER:
        count = (y == classe).sum()
        pct = count / len(y) * 100
        print(f"    {classe}: {count} ({pct:.1f}%)")

    # 2. Split estratificado
    print("\n📊 Realizando split treino/teste estratificado...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"  Treino: {len(X_train)} amostras")
    print(f"  Teste:  {len(X_test)} amostras")

    # 3. Construção do preprocessador
    print("\n🔧 Construindo preprocessador...")
    preprocessor = build_preprocessor(X_train)

    # 4. Criação do Pipeline
    print("\n🔬 Criando Pipeline (Preprocessor + LogisticRegression)...")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", get_model()),
        ]
    )

    # 5. Validação cruzada
    print("\n📈 Executando Cross-Validation (5 folds)...")
    cv_metrics = perform_cross_validation(pipeline, X_train, y_train)

    print(f"  F1 Macro:     {cv_metrics['f1_macro_mean']:.4f} ± {cv_metrics['f1_macro_std']:.4f}")
    print(f"  Recall Macro: {cv_metrics['recall_macro_mean']:.4f} ± {cv_metrics['recall_macro_std']:.4f}")
    print(f"  Recall Alto:  {cv_metrics['recall_alto_mean']:.4f} ± {cv_metrics['recall_alto_std']:.4f}")
    print(f"  Accuracy:     {cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}")

    # 6. Treinamento final com todos os dados de treino
    print("\n🏋️ Treinando modelo final...")
    pipeline.fit(X_train, y_train)

    # 7. Avaliação final no conjunto de teste
    print("\n" + "=" * 60)
    print("📋 AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print("=" * 60)
    test_metrics = evaluate_model(pipeline, X_test, y_test)

    # 8. Salvamento do modelo
    if save:
        save_model(pipeline, test_metrics, output_dir)

    # Prepara resultados
    results = {
        "model_name": "LogisticRegression",
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": X.shape[1],
    }

    print("\n" + "=" * 60)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)

    return pipeline, results


def load_model(model_path: Optional[Path] = None) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Carrega um modelo salvo.

    Args:
        model_path: Path para o arquivo do modelo (default: MODEL_DIR/model.joblib)

    Returns:
        Tuple contendo:
        - Pipeline do modelo carregado
        - Dict com metadados
    """
    if model_path is None:
        model_path = MODEL_DIR / "model.joblib"

    model_path = Path(model_path)
    metadata_path = model_path.parent / "model_metadata.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    pipeline = joblib.load(model_path)
    metadata = {}

    if metadata_path.exists():
        metadata = joblib.load(metadata_path)

    return pipeline, metadata
