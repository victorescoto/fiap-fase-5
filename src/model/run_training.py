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

import logging
from src.model.train import train_model
import pandas as pd
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add root directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


# Dataset path (adjust as needed)
DATASET_PATH = ROOT_DIR / "notebooks" / "data" / "raw.xlsx"


def main():
    """Executes the complete training pipeline."""

    logger.info("=" * 60)
    logger.info("📂 LOADING DATASET")
    logger.info("=" * 60)
    logger.info(f"   Path: {DATASET_PATH}")

    # ========================================
    # 1. DATASET LOADING (HERE!)
    # ========================================
    df = pd.read_excel(DATASET_PATH)

    logger.info(f"   Rows: {len(df)}")
    logger.info(f"   Columns: {len(df.columns)}")

    # ========================================
    # 2. MODEL TRAINING
    # ========================================
    pipeline, results = train_model(df, save=True)

    # ========================================
    # 3. FINAL SUMMARY
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Model: {results['model_name']}")


if __name__ == "__main__":
    main()
