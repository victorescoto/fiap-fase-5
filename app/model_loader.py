import json
import logging
import pickle
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent / "model"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "model.pkl"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "model_metadata.json"


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Any | None:
    """Load a serialized ML model from disk.

    Tries joblib first, then falls back to pickle.
    Returns None if the file is not found.
    """
    path = Path(path)

    if not path.exists():
        logger.warning("Model file not found at %s", path)
        return None

    try:
        model = joblib.load(path)
        logger.info("Model loaded successfully via joblib from %s", path)
        return model
    except Exception:
        logger.debug("joblib.load failed, trying pickle for %s", path)

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        logger.info("Model loaded successfully via pickle from %s", path)
        return model
    except Exception:
        logger.exception("Failed to load model from %s", path)
        return None


def load_metadata(
    path: str | Path = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Load model metadata from a JSON file.

    Returns default metadata if the file is not found or unreadable.
    """
    path = Path(path)
    defaults: dict[str, Any] = {
        "version": "unknown",
        "metrics": {},
        "features": [],
    }

    if not path.exists():
        logger.warning("Metadata file not found at %s, using defaults", path)
        return defaults

    try:
        with open(path) as f:
            metadata = json.load(f)
        logger.info("Metadata loaded from %s", path)
        return {**defaults, **metadata}
    except Exception:
        logger.exception("Failed to load metadata from %s", path)
        return defaults
