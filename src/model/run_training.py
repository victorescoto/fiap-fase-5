"""
Script to execute complete model training.

This script:
1. Loads the dataset from Excel
2. Executes training
3. Saves the model in app/model/

Usage:
    cd /Users/renatomota/Desktop/tc_challenge5/fiap-fase-5
    source .venv/bin/activate
    python -m src.model.run_training
"""

from src.model.train import train_model
import pandas as pd
import sys
from pathlib import Path

# Add root directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


# Dataset path (adjust as needed)
DATASET_PATH = Path(__file__).parent.parent.parent.parent / \
    "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

# Alternative: absolute path
# DATASET_PATH = "/Users/renatomota/Desktop/tc_challenge5/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"


def main():
    """Executes the complete training pipeline."""

    print("=" * 60)
    print("📂 LOADING DATASET")
    print("=" * 60)
    print(f"   Path: {DATASET_PATH}")

    # ========================================
    # 1. DATASET LOADING (HERE!)
    # ========================================
    df = pd.read_excel(DATASET_PATH)

    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")

    # ========================================
    # 2. MODEL TRAINING
    # ========================================
    pipeline, results = train_model(df, save=True)

    # ========================================
    # 3. FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"   Model: {results['model_name']}")


if __name__ == "__main__":
    main()
