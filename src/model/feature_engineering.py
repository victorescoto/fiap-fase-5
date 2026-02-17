"""
Feature Engineering para previsão de defasagem escolar.

Este módulo contém funções para:
- Criar a coluna target (nivel_defasagem)
- Criar features derivadas
- Remover colunas que causam data leakage
- Limpar e preparar o DataFrame para treinamento
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


# Colunas que causam data leakage (contêm informação do target ou são identificadores)
LEAKAGE_COLUMNS = [
    # Identificadores pessoais
    "RA",
    "Nome",
    "Turma",
    # Colunas relacionadas diretamente ao target
    "Defas",  # É usada para criar o target
    "Fase ideal",  # Relacionada diretamente à defasagem
    # Avaliadores (podem conter viés ou informação futura)
    "Avaliador1",
    "Avaliador2",
    "Avaliador3",
    "Avaliador4",
    # Recomendações dos avaliadores (decisões baseadas em análise completa)
    "Rec Av1",
    "Rec Av2",
    "Rec Av3",
    "Rec Av4",
    # Colunas de destaque (derivadas de análise)
    "Destaque IEG",
    "Destaque IDA",
    "Destaque IPV",
]

# Mapeamento das pedras para valores numéricos (ordem de progressão)
PEDRA_MAPPING = {
    "Quartzo": 1,
    "Ágata": 2,
    "Ametista": 3,
    "Topázio": 4,
}

# Limites para classificação da defasagem
# Baseado na distribuição dos dados:
# - Valores >= 0: aluno na fase ideal ou adiantado (baixo)
# - Valores -1 ou -2: defasagem moderada (medio)
# - Valores <= -3: defasagem severa (alto)
DEFASAGEM_THRESHOLDS = {
    "baixo": (0, float("inf")),  # Defas >= 0
    "medio": (-2, -1),  # Defas entre -2 e -1
    "alto": (float("-inf"), -3),  # Defas <= -3
}


def create_target_column(df: pd.DataFrame, column: str = "Defas") -> pd.Series:
    """
    Cria a coluna target 'nivel_defasagem' baseada na coluna de defasagem.

    A classificação é feita da seguinte forma:
    - "baixo": defasagem >= 0 (aluno na fase ideal ou adiantado)
    - "medio": defasagem entre -1 e -2
    - "alto": defasagem <= -3 (defasagem severa)

    Args:
        df: DataFrame com a coluna de defasagem
        column: Nome da coluna de defasagem (default: "Defas")

    Returns:
        pd.Series: Série com os níveis de defasagem ("baixo", "medio", "alto")
    """
    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada no DataFrame")

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
    Converte colunas de 'Pedra' para valores numéricos.

    Args:
        df: DataFrame com colunas Pedra

    Returns:
        DataFrame com colunas Pedra convertidas para numérico
    """
    pedra_cols = [col for col in df.columns if col.startswith("Pedra")]

    for col in pedra_cols:
        if col in df.columns:
            df[f"{col}_encoded"] = df[col].map(PEDRA_MAPPING)

    return df


def _create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features temporais baseadas na progressão do aluno.

    Features criadas:
    - tempo_no_programa: Anos desde o ingresso até 2022
    - idade_ingresso: Idade quando ingressou no programa
    - pedra_evolucao_20_21: Diferença entre pedras 2020 e 2021
    - pedra_evolucao_21_22: Diferença entre pedras 2021 e 2022
    - pedra_evolucao_total: Evolução total nas pedras

    Args:
        df: DataFrame original

    Returns:
        DataFrame com novas features temporais
    """
    df = df.copy()

    # Tempo no programa (considerando dados de 2022)
    if "Ano ingresso" in df.columns:
        df["tempo_no_programa"] = 2022 - df["Ano ingresso"]

    # Idade quando ingressou
    if "Ano nasc" in df.columns and "Ano ingresso" in df.columns:
        df["idade_ingresso"] = df["Ano ingresso"] - df["Ano nasc"]

    # Evolução nas pedras (se as colunas encoded existirem)
    if "Pedra 20_encoded" in df.columns and "Pedra 21_encoded" in df.columns:
        df["pedra_evolucao_20_21"] = df["Pedra 21_encoded"] - df["Pedra 20_encoded"]

    if "Pedra 21_encoded" in df.columns and "Pedra 22_encoded" in df.columns:
        df["pedra_evolucao_21_22"] = df["Pedra 22_encoded"] - df["Pedra 21_encoded"]

    if "Pedra 20_encoded" in df.columns and "Pedra 22_encoded" in df.columns:
        df["pedra_evolucao_total"] = df["Pedra 22_encoded"] - df["Pedra 20_encoded"]

    return df


def _create_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features baseadas em performance acadêmica.

    Features criadas:
    - media_disciplinas: Média entre Matemática, Português e Inglês
    - desempenho_geral: Média ponderada dos indicadores
    - ratio_indicadores: Razão entre diferentes indicadores
    - num_avaliacoes_positivas: Contagem de avaliações positivas

    Args:
        df: DataFrame original

    Returns:
        DataFrame com novas features de performance
    """
    df = df.copy()

    # Média das disciplinas
    disciplinas = ["Matem", "Portug", "Inglês"]
    cols_presentes = [col for col in disciplinas if col in df.columns]
    if cols_presentes:
        df["media_disciplinas"] = df[cols_presentes].mean(axis=1, skipna=True)
        df["std_disciplinas"] = df[cols_presentes].std(axis=1, skipna=True)
        df["min_disciplina"] = df[cols_presentes].min(axis=1, skipna=True)
        df["max_disciplina"] = df[cols_presentes].max(axis=1, skipna=True)

    # Indicadores compostos
    indicadores = ["IAA", "IEG", "IPS", "IDA", "IPV", "IAN"]
    ind_presentes = [col for col in indicadores if col in df.columns]
    if ind_presentes:
        df["media_indicadores"] = df[ind_presentes].mean(axis=1, skipna=True)
        df["std_indicadores"] = df[ind_presentes].std(axis=1, skipna=True)

    # Razão INDE vs média de indicadores
    if "INDE 22" in df.columns and "media_indicadores" in df.columns:
        df["ratio_inde_indicadores"] = df["INDE 22"] / (
            df["media_indicadores"] + 1e-6
        )  # Evita divisão por zero

    # Comparação entre indicadores específicos
    if "IAA" in df.columns and "IDA" in df.columns:
        df["diff_iaa_ida"] = df["IAA"] - df["IDA"]

    if "IEG" in df.columns and "IPS" in df.columns:
        df["diff_ieg_ips"] = df["IEG"] - df["IPS"]

    return df


def _create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features relacionadas ao engajamento do aluno.

    Features criadas:
    - indicado_bin: Versão binária da coluna Indicado
    - atingiu_pv_bin: Versão binária da coluna Atingiu PV
    - psicologia_flag: Flag indicando se requer avaliação psicológica

    Args:
        df: DataFrame original

    Returns:
        DataFrame com novas features de engajamento
    """
    df = df.copy()

    # Conversão de colunas binárias
    if "Indicado" in df.columns:
        df["indicado_bin"] = (df["Indicado"].str.lower() == "sim").astype(int)

    if "Atingiu PV" in df.columns:
        df["atingiu_pv_bin"] = (df["Atingiu PV"].str.lower() == "sim").astype(int)

    # Flag de psicologia
    if "Rec Psicologia" in df.columns:
        df["psicologia_requer_avaliacao"] = (
            df["Rec Psicologia"].str.lower().str.contains("requer", na=False)
        ).astype(int)

    return df


def _remove_leakage_columns(
    df: pd.DataFrame, additional_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Remove colunas que podem causar data leakage.

    Args:
        df: DataFrame original
        additional_columns: Lista adicional de colunas para remover

    Returns:
        DataFrame sem as colunas de leakage
    """
    columns_to_remove = LEAKAGE_COLUMNS.copy()

    if additional_columns:
        columns_to_remove.extend(additional_columns)

    # Remove colunas originais de Pedra (mantém apenas as encoded)
    pedra_cols = [col for col in df.columns if col.startswith("Pedra") and "_encoded" not in col]
    columns_to_remove.extend(pedra_cols)

    # Filtra apenas colunas que existem no DataFrame
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]

    return df.drop(columns=columns_to_remove, errors="ignore")


def build_features(
    df: pd.DataFrame, include_target: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Função principal de feature engineering.

    Executa todas as transformações de features:
    1. Cria a coluna target (nivel_defasagem)
    2. Codifica colunas de Pedra
    3. Cria features temporais
    4. Cria features de performance
    5. Cria features de engajamento
    6. Remove colunas de leakage

    Args:
        df: DataFrame original com dados brutos
        include_target: Se True, retorna também a série target

    Returns:
        Tuple contendo:
        - DataFrame processado com features
        - Series com target (se include_target=True, senão None)
    """
    df = df.copy()

    # Cria target antes de qualquer processamento
    target = None
    if include_target:
        if "Defas" not in df.columns:
            raise ValueError("Coluna 'Defas' necessária para criar target")
        target = create_target_column(df)

    # Pipeline de feature engineering
    df = _encode_pedra_columns(df)
    df = _create_temporal_features(df)
    df = _create_performance_features(df)
    df = _create_engagement_features(df)

    # Remove colunas de leakage
    df = _remove_leakage_columns(df)

    # Remove colunas categóricas originais que foram processadas
    # (Indicado, Atingiu PV, Rec Psicologia - já criamos versões binárias)
    cols_processed = ["Indicado", "Atingiu PV", "Rec Psicologia"]
    df = df.drop(columns=[c for c in cols_processed if c in df.columns], errors="ignore")

    return df, target


def get_feature_names() -> dict:
    """
    Retorna dicionário com categorização das features.

    Returns:
        Dict com categorias de features e suas descrições
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
