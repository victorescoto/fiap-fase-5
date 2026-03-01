"""Tests for src.model.preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.model.preprocessing import (
    build_preprocessor,
    get_feature_names_from_preprocessor,
    identify_column_types,
    validate_preprocessor,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """DataFrame with numeric, categorical and missing values."""
    return pd.DataFrame(
        {
            "age": [10, 12, np.nan, 14, 11],
            "score": [7.5, 6.0, 5.0, np.nan, 8.0],
            "group": ["A", "B", "A", None, "B"],
            "flag": [1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def numeric_only_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [4, 5, 6],
        }
    )


# ------------------------------------------------------------------
# identify_column_types
# ------------------------------------------------------------------


class TestIdentifyColumnTypes:
    def test_mixed_types(self, mixed_df: pd.DataFrame) -> None:
        numeric, categorical = identify_column_types(mixed_df)
        assert "age" in numeric
        assert "score" in numeric
        assert "flag" in numeric
        assert "group" in categorical

    def test_numeric_only(self, numeric_only_df: pd.DataFrame) -> None:
        numeric, categorical = identify_column_types(numeric_only_df)
        assert len(numeric) == 2
        assert len(categorical) == 0

    def test_empty_dataframe(self) -> None:
        numeric, categorical = identify_column_types(pd.DataFrame())
        assert numeric == []
        assert categorical == []


# ------------------------------------------------------------------
# build_preprocessor
# ------------------------------------------------------------------


class TestBuildPreprocessor:
    def test_returns_column_transformer(self, mixed_df: pd.DataFrame) -> None:
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor(mixed_df)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_transform_shape(self, mixed_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(mixed_df)
        result = preprocessor.fit_transform(mixed_df)
        assert result.shape[0] == len(mixed_df)
        # 3 numeric + 2 one-hot for group (A, B) minus 1 for drop=if_binary = 4 cols
        assert result.shape[1] >= 3

    def test_handles_missing_values(self, mixed_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(mixed_df)
        result = preprocessor.fit_transform(mixed_df)
        assert not np.isnan(result).any(), "Preprocessor should impute all NaN"

    def test_with_explicit_columns(self, mixed_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(
            mixed_df, numeric_cols=["age"], categorical_cols=["group"]
        )
        result = preprocessor.fit_transform(mixed_df)
        assert result.shape[0] == len(mixed_df)

    def test_numeric_only_df(self, numeric_only_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(numeric_only_df)
        result = preprocessor.fit_transform(numeric_only_df)
        assert result.shape == (3, 2)


# ------------------------------------------------------------------
# get_feature_names_from_preprocessor
# ------------------------------------------------------------------


class TestGetFeatureNames:
    def test_returns_list_of_strings(self, mixed_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(mixed_df)
        preprocessor.fit(mixed_df)
        names = get_feature_names_from_preprocessor(preprocessor, mixed_df)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_names_count_matches_transform(self, mixed_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(mixed_df)
        result = preprocessor.fit_transform(mixed_df)
        names = get_feature_names_from_preprocessor(preprocessor, mixed_df)
        assert len(names) == result.shape[1]


# ------------------------------------------------------------------
# validate_preprocessor
# ------------------------------------------------------------------


class TestValidatePreprocessor:
    def test_returns_expected_keys(self, mixed_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(mixed_df)
        info = validate_preprocessor(preprocessor, mixed_df)
        assert "total_columns" in info
        assert "numeric_columns" in info
        assert "categorical_columns" in info
        assert "missing_total" in info

    def test_counts_missing_values(self, mixed_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(mixed_df)
        info = validate_preprocessor(preprocessor, mixed_df)
        assert info["missing_total"] > 0
