"""Tests for src.model.feature_engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.model.feature_engineering import (
    LEAKAGE_COLUMNS,
    PEDRA_MAPPING,
    build_features,
    create_target_column,
    get_feature_names,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame that exercises the full feature-engineering pipeline."""
    return pd.DataFrame(
        {
            # Identifiers (should be dropped)
            "RA": [1, 2, 3, 4, 5],
            "Nome": ["A", "B", "C", "D", "E"],
            "Turma": ["T1", "T1", "T2", "T2", "T3"],
            # Target source
            "Defas": [0, -1, -2, -3, 1],
            "Fase ideal": [5, 5, 5, 5, 5],
            # Pedra columns (should be dropped as leakage)
            "Pedra 20": ["Quartzo", "Ágata", "Ametista", "Topázio", "Quartzo"],
            "Pedra 21": ["Ágata", "Ametista", "Topázio", "Topázio", "Ágata"],
            "Pedra 22": ["Ametista", "Topázio", "Topázio", "Topázio", "Ametista"],
            # Numeric features
            "Fase": [1, 2, 3, 4, 5],
            "Ano nasc": [2010, 2009, 2008, 2007, 2011],
            "Idade 22": [12, 13, 14, 15, 11],
            "Ano ingresso": [2018, 2017, 2016, 2015, 2020],
            "INDE 22": [7.5, 6.0, 5.0, 3.0, 8.0],
            "Cg": [7.0, 6.0, 5.0, 4.0, 8.0],
            "Cf": [8.0, 7.0, 6.0, 5.0, 9.0],
            "Ct": [6.0, 5.0, 4.0, 3.0, 7.0],
            "Nº Av": [4, 3, 2, 1, 4],
            "IAA": [7.0, 6.0, 5.0, 4.0, 8.0],
            "IEG": [6.5, 5.5, 4.5, 3.5, 7.5],
            "IPS": [7.0, 6.0, 5.0, 4.0, 8.0],
            "IDA": [6.0, 5.0, 4.0, 3.0, 7.0],
            "IPV": [8.0, 7.0, 6.0, 5.0, 9.0],
            "IAN": [5.0, 4.0, 3.0, 2.0, 6.0],
            "Matem": [7.0, 6.0, 5.0, 4.0, 8.0],
            "Portug": [8.0, 7.0, 6.0, 5.0, 9.0],
            "Inglês": [6.0, 5.0, 4.0, 3.0, 7.0],
            # Evaluator columns (leakage)
            "Avaliador1": ["Av1", "Av2", "Av3", "Av4", "Av1"],
            "Rec Av1": ["ok", "ok", "ok", "ok", "ok"],
            # Highlight columns (leakage)
            "Destaque IEG": ["sim", "nao", "sim", "nao", "sim"],
            "Destaque IDA": ["nao", "sim", "nao", "sim", "nao"],
            "Destaque IPV": ["sim", "sim", "nao", "nao", "sim"],
            # Engagement columns
            "Indicado": ["Sim", "Não", "Sim", "Não", "Sim"],
            "Atingiu PV": ["Sim", "Sim", "Não", "Não", "Sim"],
            "Rec Psicologia": ["requer avaliação", "ok", "requer", "ok", "ok"],
        }
    )


# ------------------------------------------------------------------
# create_target_column
# ------------------------------------------------------------------


class TestCreateTargetColumn:
    def test_baixo_for_zero_or_positive(self, sample_df: pd.DataFrame) -> None:
        target = create_target_column(sample_df)
        # Defas=0 and Defas=1 → baixo
        assert target.iloc[0] == "baixo"
        assert target.iloc[4] == "baixo"

    def test_medio_for_minus_one_and_minus_two(self, sample_df: pd.DataFrame) -> None:
        target = create_target_column(sample_df)
        assert target.iloc[1] == "medio"  # Defas = -1
        assert target.iloc[2] == "medio"  # Defas = -2

    def test_alto_for_minus_three_or_lower(self, sample_df: pd.DataFrame) -> None:
        target = create_target_column(sample_df)
        assert target.iloc[3] == "alto"  # Defas = -3

    def test_nan_input_returns_nan(self) -> None:
        df = pd.DataFrame({"Defas": [0, np.nan, -3]})
        target = create_target_column(df)
        assert target.iloc[0] == "baixo"
        assert pd.isna(target.iloc[1])
        assert target.iloc[2] == "alto"

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(ValueError, match="Column.*not found"):
            create_target_column(df)

    def test_custom_column_name(self) -> None:
        df = pd.DataFrame({"MyDelay": [0, -1, -4]})
        target = create_target_column(df, column="MyDelay")
        assert list(target) == ["baixo", "medio", "alto"]


# ------------------------------------------------------------------
# build_features
# ------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_dataframe_and_series(self, sample_df: pd.DataFrame) -> None:
        X, y = build_features(sample_df, include_target=True)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == len(sample_df)

    def test_no_target_when_disabled(self, sample_df: pd.DataFrame) -> None:
        X, y = build_features(sample_df, include_target=False)
        assert y is None

    def test_leakage_columns_removed(self, sample_df: pd.DataFrame) -> None:
        X, _ = build_features(sample_df)
        remaining = set(X.columns)
        for col in LEAKAGE_COLUMNS:
            assert col not in remaining, f"Leakage column '{col}' still present"

    def test_pedra_encoded_columns_removed(self, sample_df: pd.DataFrame) -> None:
        """Pedra encoded columns are leakage and must not be in the output."""
        X, _ = build_features(sample_df)
        for col in X.columns:
            assert "Pedra" not in col, f"Pedra-related column '{col}' still present"
            assert "pedra_evolucao" not in col, f"Pedra evolution '{col}' still present"

    def test_inde_22_removed(self, sample_df: pd.DataFrame) -> None:
        """INDE 22 is leakage and must be removed."""
        X, _ = build_features(sample_df)
        assert "INDE 22" not in X.columns

    def test_original_categoricals_dropped(self, sample_df: pd.DataFrame) -> None:
        X, _ = build_features(sample_df)
        for col in ["Indicado", "Atingiu PV", "Rec Psicologia"]:
            assert col not in X.columns

    def test_derived_features_created(self, sample_df: pd.DataFrame) -> None:
        X, _ = build_features(sample_df)
        expected = [
            "tempo_no_programa",
            "idade_ingresso",
            "media_disciplinas",
            "indicado_bin",
            "atingiu_pv_bin",
        ]
        for feat in expected:
            assert feat in X.columns, f"Expected feature '{feat}' missing"

    def test_no_nan_target_labels(self, sample_df: pd.DataFrame) -> None:
        _, y = build_features(sample_df)
        assert y.notna().all()


# ------------------------------------------------------------------
# PEDRA_MAPPING
# ------------------------------------------------------------------


class TestPedraMapping:
    def test_order(self) -> None:
        assert PEDRA_MAPPING["Quartzo"] < PEDRA_MAPPING["Ágata"]
        assert PEDRA_MAPPING["Ágata"] < PEDRA_MAPPING["Ametista"]
        assert PEDRA_MAPPING["Ametista"] < PEDRA_MAPPING["Topázio"]


# ------------------------------------------------------------------
# get_feature_names
# ------------------------------------------------------------------


class TestGetFeatureNames:
    def test_returns_dict_with_expected_keys(self) -> None:
        names = get_feature_names()
        assert "temporais" in names
        assert "performance" in names
        assert "engajamento" in names
