"""
Script para executar o treinamento completo do modelo.

Este script:
1. Carrega o dataset do Excel
2. Executa o treinamento
3. Salva o modelo em app/model/

Uso:
    cd /Users/renatomota/Desktop/tc_challenge5/fiap-fase-5
    source .venv/bin/activate
    python -m src.model.run_training
"""

from src.model.train import train_model
import pandas as pd
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


# Caminho do dataset (ajuste conforme necessário)
DATASET_PATH = Path(__file__).parent.parent.parent.parent / \
    "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

# Alternativa: caminho absoluto
# DATASET_PATH = "/Users/renatomota/Desktop/tc_challenge5/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"


def main():
    """Executa o pipeline completo de treinamento."""

    print("=" * 60)
    print("📂 CARREGANDO DATASET")
    print("=" * 60)
    print(f"   Caminho: {DATASET_PATH}")

    # ========================================
    # 1. CARREGAMENTO DO DATASET (AQUI!)
    # ========================================
    df = pd.read_excel(DATASET_PATH)

    print(f"   Linhas: {len(df)}")
    print(f"   Colunas: {len(df.columns)}")

    # ========================================
    # 2. TREINAMENTO DO MODELO
    # ========================================
    pipeline, results = train_model(df, save=True)

    # ========================================
    # 3. RESUMO FINAL
    # ========================================
    print("\n" + "=" * 60)
    print("📊 RESUMO FINAL")
    print("=" * 60)
    print(f"   Modelo: {results['model_name']}")


if __name__ == "__main__":
    main()
