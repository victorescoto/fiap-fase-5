"""
Pré-processamento de dados para o modelo de defasagem escolar.

Este módulo contém funções para:
- Identificar automaticamente colunas numéricas e categóricas
- Criar transformadores para cada tipo de coluna
- Construir um ColumnTransformer completo para o pipeline
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def identify_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifica automaticamente colunas numéricas e categóricas.

    Regras:
    - Colunas com dtype numérico (int, float) -> numéricas
    - Colunas com dtype object ou category -> categóricas
    - Colunas booleanas -> numéricas (já são 0/1)

    Args:
        X: DataFrame com as features

    Returns:
        Tuple contendo:
        - Lista de nomes de colunas numéricas
        - Lista de nomes de colunas categóricas
    """
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        dtype = X[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            categorical_cols.append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            numeric_cols.append(col)
        else:
            # Por segurança, trata como categórica
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_numeric_transformer() -> Pipeline:
    """
    Constrói o pipeline de transformação para colunas numéricas.

    Etapas:
    1. SimpleImputer com estratégia median (robusto a outliers)
    2. StandardScaler para normalização

    Returns:
        Pipeline de transformação numérica
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_transformer() -> Pipeline:
    """
    Constrói o pipeline de transformação para colunas categóricas.

    Etapas:
    1. SimpleImputer com estratégia most_frequent
    2. OneHotEncoder com handle_unknown='ignore' para lidar com categorias não vistas

    Returns:
        Pipeline de transformação categórica
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",  # Evita multicolinearidade para variáveis binárias
                ),
            ),
        ]
    )


def build_preprocessor(
    X: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer completo para pré-processamento.

    Se as listas de colunas não forem fornecidas, identifica automaticamente
    os tipos de colunas baseado nos dtypes do DataFrame.

    O preprocessor aplica:
    - Para numéricas: SimpleImputer(median) + StandardScaler
    - Para categóricas: SimpleImputer(most_frequent) + OneHotEncoder

    Args:
        X: DataFrame com as features
        numeric_cols: Lista opcional de colunas numéricas
        categorical_cols: Lista opcional de colunas categóricas

    Returns:
        ColumnTransformer configurado para pré-processamento
    """
    # Identifica tipos de colunas se não fornecidos
    if numeric_cols is None or categorical_cols is None:
        auto_numeric, auto_categorical = identify_column_types(X)
        numeric_cols = numeric_cols if numeric_cols is not None else auto_numeric
        categorical_cols = categorical_cols if categorical_cols is not None else auto_categorical

    # Filtra apenas colunas que existem no DataFrame
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    categorical_cols = [col for col in categorical_cols if col in X.columns]

    # Lista de transformadores
    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", build_numeric_transformer(), numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", build_categorical_transformer(), categorical_cols)
        )

    # Cria o ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Remove colunas não especificadas
        verbose_feature_names_out=True,  # Mantém nomes das features
    )

    return preprocessor


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer, X: pd.DataFrame
) -> List[str]:
    """
    Extrai os nomes das features após transformação.

    Útil para interpretação do modelo e análise de importância de features.

    Args:
        preprocessor: ColumnTransformer já treinado (fit)
        X: DataFrame usado no fit

    Returns:
        Lista com nomes das features transformadas
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback para versões mais antigas do sklearn
        feature_names = []

        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue

            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(columns)
            else:
                names = columns

            feature_names.extend(names)

        return feature_names


def validate_preprocessor(preprocessor: ColumnTransformer, X: pd.DataFrame) -> dict:
    """
    Valida o preprocessor e retorna informações sobre as transformações.

    Args:
        preprocessor: ColumnTransformer (não precisa estar treinado)
        X: DataFrame para validação

    Returns:
        Dict com informações sobre o preprocessor
    """
    numeric_cols, categorical_cols = identify_column_types(X)

    info = {
        "total_columns": len(X.columns),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "numeric_column_names": numeric_cols,
        "categorical_column_names": categorical_cols,
        "missing_values": X.isnull().sum().to_dict(),
        "missing_total": X.isnull().sum().sum(),
    }

    return info
