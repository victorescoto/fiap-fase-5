"""
Data preprocessing for the school delay prediction model.

This module contains functions to:
- Automatically identify numeric and categorical columns
- Create transformers for each column type
- Build a complete ColumnTransformer for the pipeline
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
    Automatically identifies numeric and categorical columns.

    Rules:
    - Columns with numeric dtype (int, float) -> numeric
    - Columns with object or category dtype -> categorical
    - Boolean columns -> numeric (already 0/1)

    Args:
        X: DataFrame with features

    Returns:
        Tuple containing:
        - List of numeric column names
        - List of categorical column names
    """
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        dtype = X[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(
            dtype
        ):
            categorical_cols.append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            numeric_cols.append(col)
        else:
            # For safety, treat as categorical
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_numeric_transformer() -> Pipeline:
    """
    Builds the transformation pipeline for numeric columns.

    Steps:
    1. SimpleImputer with median strategy (robust to outliers)
    2. StandardScaler for normalization

    Returns:
        Numeric transformation pipeline
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_transformer() -> Pipeline:
    """
    Builds the transformation pipeline for categorical columns.

    Steps:
    1. SimpleImputer with most_frequent strategy
    2. OneHotEncoder with handle_unknown='ignore' to handle unseen categories

    Returns:
        Categorical transformation pipeline
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",  # Avoids multicollinearity for binary variables
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
    Builds the complete ColumnTransformer for preprocessing.

    If column lists are not provided, automatically identifies
    column types based on DataFrame dtypes.

    The preprocessor applies:
    - For numeric: SimpleImputer(median) + StandardScaler
    - For categorical: SimpleImputer(most_frequent) + OneHotEncoder

    Args:
        X: DataFrame with features
        numeric_cols: Optional list of numeric columns
        categorical_cols: Optional list of categorical columns

    Returns:
        ColumnTransformer configured for preprocessing
    """
    # Identify column types if not provided
    if numeric_cols is None or categorical_cols is None:
        auto_numeric, auto_categorical = identify_column_types(X)
        numeric_cols = numeric_cols if numeric_cols is not None else auto_numeric
        categorical_cols = (
            categorical_cols if categorical_cols is not None else auto_categorical
        )

    # Filter only columns that exist in the DataFrame
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    categorical_cols = [col for col in categorical_cols if col in X.columns]

    # List of transformers
    transformers = []

    if numeric_cols:
        transformers.append(("numeric", build_numeric_transformer(), numeric_cols))

    if categorical_cols:
        transformers.append(
            ("categorical", build_categorical_transformer(), categorical_cols)
        )

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop unspecified columns
        verbose_feature_names_out=True,  # Keep feature names
    )

    return preprocessor


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer, X: pd.DataFrame
) -> List[str]:
    """
    Extracts feature names after transformation.

    Useful for model interpretation and feature importance analysis.

    Args:
        preprocessor: Already trained (fit) ColumnTransformer
        X: DataFrame used in fit

    Returns:
        List with transformed feature names
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn versions
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
    Validates the preprocessor and returns information about transformations.

    Args:
        preprocessor: ColumnTransformer (does not need to be trained)
        X: DataFrame for validation

    Returns:
        Dict with information about the preprocessor
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
