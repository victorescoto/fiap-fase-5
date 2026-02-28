"""
Feature Engineering for school delay prediction.

This module contains functions to:
- Create the target column (nivel_defasagem)
- Create derived features
- Remove columns that cause data leakage
- Clean and prepare the DataFrame for training
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


# Columns that cause data leakage (contain target information or are identifiers)
LEAKAGE_COLUMNS = [
    # Personal identifiers
    "RA",
    "Nome",
    "Turma",
    # Columns directly related to the target
    "Defas",  # Used to create the target
    "Fase ideal",  # Directly related to delay
    # Evaluators (may contain bias or future information)
    "Avaliador1",
    "Avaliador2",
    "Avaliador3",
    "Avaliador4",
    # Evaluator recommendations (decisions based on complete analysis)
    "Rec Av1",
    "Rec Av2",
    "Rec Av3",
    "Rec Av4",
    # Highlight columns (derived from analysis)
    "Destaque IEG",
    "Destaque IDA",
    "Destaque IPV",
]

# Mapping stones to numeric values (progression order)
PEDRA_MAPPING = {
    "Quartzo": 1,
    "Ágata": 2,
    "Ametista": 3,
    "Topázio": 4,
}

# Thresholds for delay classification
# Based on data distribution:
# - Values >= 0: student at ideal phase or ahead (baixo)
# - Values -1 or -2: moderate delay (medio)
# - Values <= -3: severe delay (alto)
DEFASAGEM_THRESHOLDS = {
    "baixo": (0, float("inf")),  # Defas >= 0
    "medio": (-2, -1),  # Defas between -2 and -1
    "alto": (float("-inf"), -3),  # Defas <= -3
}


def create_target_column(df: pd.DataFrame, column: str = "Defas") -> pd.Series:
    """
    Creates the target column 'nivel_defasagem' based on the delay column.

    The classification is done as follows:
    - "baixo": delay >= 0 (student at ideal phase or ahead)
    - "medio": delay between -1 and -2
    - "alto": delay <= -3 (severe delay)

    Args:
        df: DataFrame with the delay column
        column: Name of the delay column (default: "Defas")

    Returns:
        pd.Series: Series with delay levels ("baixo", "medio", "alto")
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    def classify_defasagem(value: int) -> str:
        if pd.isna(value):
            return np.nan
        if value >= 0:
            return "baixo"
        elif value >= -2:
            return "medio"
        else:
            return "alto"

    return df[column].apply(classify_defasagem)


def _encode_pedra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts 'Pedra' columns to numeric values.

    Args:
        df: DataFrame with Pedra columns

    Returns:
        DataFrame with Pedra columns converted to numeric
    """
    pedra_cols = [col for col in df.columns if col.startswith("Pedra")]

    for col in pedra_cols:
        if col in df.columns:
            df[f"{col}_encoded"] = df[col].map(PEDRA_MAPPING)

    return df


def _create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates temporal features based on student progression.

    Features created:
    - tempo_no_programa: Years since entry until 2022
    - idade_ingresso: Age when joined the program
    - pedra_evolucao_20_21: Difference between 2020 and 2021 stones
    - pedra_evolucao_21_22: Difference between 2021 and 2022 stones
    - pedra_evolucao_total: Total evolution in stones

    Args:
        df: Original DataFrame

    Returns:
        DataFrame with new temporal features
    """
    df = df.copy()

    # Time in program (considering 2022 data)
    if "Ano ingresso" in df.columns:
        df["tempo_no_programa"] = 2022 - df["Ano ingresso"]

    # Age when joined
    if "Ano nasc" in df.columns and "Ano ingresso" in df.columns:
        df["idade_ingresso"] = df["Ano ingresso"] - df["Ano nasc"]

    # Evolution in stones (if encoded columns exist)
    if "Pedra 20_encoded" in df.columns and "Pedra 21_encoded" in df.columns:
        df["pedra_evolucao_20_21"] = df["Pedra 21_encoded"] - df["Pedra 20_encoded"]

    if "Pedra 21_encoded" in df.columns and "Pedra 22_encoded" in df.columns:
        df["pedra_evolucao_21_22"] = df["Pedra 22_encoded"] - df["Pedra 21_encoded"]

    if "Pedra 20_encoded" in df.columns and "Pedra 22_encoded" in df.columns:
        df["pedra_evolucao_total"] = df["Pedra 22_encoded"] - df["Pedra 20_encoded"]

    return df


def _create_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features based on academic performance.

    Features created:
    - media_disciplinas: Average between Math, Portuguese and English
    - desempenho_geral: Weighted average of indicators
    - ratio_indicadores: Ratio between different indicators
    - num_avaliacoes_positivas: Count of positive evaluations

    Args:
        df: Original DataFrame

    Returns:
        DataFrame with new performance features
    """
    df = df.copy()

    # Average of subjects
    disciplinas = ["Matem", "Portug", "Inglês"]
    cols_presentes = [col for col in disciplinas if col in df.columns]
    if cols_presentes:
        df["media_disciplinas"] = df[cols_presentes].mean(axis=1, skipna=True)
        df["std_disciplinas"] = df[cols_presentes].std(axis=1, skipna=True)
        df["min_disciplina"] = df[cols_presentes].min(axis=1, skipna=True)
        df["max_disciplina"] = df[cols_presentes].max(axis=1, skipna=True)

    # Composite indicators
    indicadores = ["IAA", "IEG", "IPS", "IDA", "IPV", "IAN"]
    ind_presentes = [col for col in indicadores if col in df.columns]
    if ind_presentes:
        df["media_indicadores"] = df[ind_presentes].mean(axis=1, skipna=True)
        df["std_indicadores"] = df[ind_presentes].std(axis=1, skipna=True)

    # INDE ratio vs average of indicators
    if "INDE 22" in df.columns and "media_indicadores" in df.columns:
        df["ratio_inde_indicadores"] = df["INDE 22"] / (
            df["media_indicadores"] + 1e-6
        )  # Avoid division by zero

    # Comparison between specific indicators
    if "IAA" in df.columns and "IDA" in df.columns:
        df["diff_iaa_ida"] = df["IAA"] - df["IDA"]

    if "IEG" in df.columns and "IPS" in df.columns:
        df["diff_ieg_ips"] = df["IEG"] - df["IPS"]

    return df


def _create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features related to student engagement.

    Features created:
    - indicado_bin: Binary version of Indicado column
    - atingiu_pv_bin: Binary version of Atingiu PV column
    - psicologia_flag: Flag indicating if requires psychological evaluation

    Args:
        df: Original DataFrame

    Returns:
        DataFrame with new engagement features
    """
    df = df.copy()

    # Conversion of binary columns
    if "Indicado" in df.columns:
        df["indicado_bin"] = (df["Indicado"].str.lower() == "sim").astype(int)

    if "Atingiu PV" in df.columns:
        df["atingiu_pv_bin"] = (df["Atingiu PV"].str.lower() == "sim").astype(int)

    # Psychology flag
    if "Rec Psicologia" in df.columns:
        df["psicologia_requer_avaliacao"] = (
            df["Rec Psicologia"].str.lower().str.contains("requer", na=False)
        ).astype(int)

    return df


def _remove_leakage_columns(
    df: pd.DataFrame, additional_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Removes columns that may cause data leakage.

    Args:
        df: Original DataFrame
        additional_columns: Additional list of columns to remove

    Returns:
        DataFrame without leakage columns
    """
    columns_to_remove = LEAKAGE_COLUMNS.copy()

    if additional_columns:
        columns_to_remove.extend(additional_columns)

    # Remove original Pedra columns (keep only encoded ones)
    pedra_cols = [col for col in df.columns if col.startswith("Pedra") and "_encoded" not in col]
    columns_to_remove.extend(pedra_cols)

    # Filter only columns that exist in the DataFrame
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]

    return df.drop(columns=columns_to_remove, errors="ignore")


def build_features(
    df: pd.DataFrame, include_target: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Main feature engineering function.

    Executes all feature transformations:
    1. Creates target column (nivel_defasagem)
    2. Encodes Pedra columns
    3. Creates temporal features
    4. Creates performance features
    5. Creates engagement features
    6. Removes leakage columns

    Args:
        df: Original DataFrame with raw data
        include_target: If True, also returns target series

    Returns:
        Tuple containing:
        - Processed DataFrame with features
        - Series with target (if include_target=True, otherwise None)
    """
    df = df.copy()

    # Create target before any processing
    target = None
    if include_target:
        if "Defas" not in df.columns:
            raise ValueError("Column 'Defas' required to create target")
        target = create_target_column(df)

    # Feature engineering pipeline
    df = _encode_pedra_columns(df)
    df = _create_temporal_features(df)
    df = _create_performance_features(df)
    df = _create_engagement_features(df)

    # Remove leakage columns
    df = _remove_leakage_columns(df)

    # Remove original categorical columns that were processed
    # (Indicado, Atingiu PV, Rec Psicologia - we already created binary versions)
    cols_processed = ["Indicado", "Atingiu PV", "Rec Psicologia"]
    df = df.drop(columns=[c for c in cols_processed if c in df.columns], errors="ignore")

    return df, target


def get_feature_names() -> dict:
    """
    Returns dictionary with feature categorization.

    Returns:
        Dict with feature categories and their descriptions
    """
    return {
        "temporais": [
            "tempo_no_programa",
            "idade_ingresso",
            "pedra_evolucao_20_21",
            "pedra_evolucao_21_22",
            "pedra_evolucao_total",
        ],
        "performance": [
            "media_disciplinas",
            "std_disciplinas",
            "min_disciplina",
            "max_disciplina",
            "media_indicadores",
            "std_indicadores",
            "ratio_inde_indicadores",
            "diff_iaa_ida",
            "diff_ieg_ips",
        ],
        "engajamento": [
            "indicado_bin",
            "atingiu_pv_bin",
            "psicologia_requer_avaliacao",
        ],
        "pedras_encoded": [
            "Pedra 20_encoded",
            "Pedra 21_encoded",
            "Pedra 22_encoded",
        ],
    }
