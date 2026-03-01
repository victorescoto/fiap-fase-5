"""Tests for src.model.train module."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.model.train import (
    CLASS_ORDER,
    create_custom_scorers,
    get_model,
    save_model,
    train_model,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame suitable for training."""
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame(
        {
            "RA": range(n),
            "Nome": [f"Student_{i}" for i in range(n)],
            "Turma": rng.choice(["T1", "T2"], n),
            "Defas": rng.choice([0, -1, -2, -3, 1], n, p=[0.3, 0.25, 0.25, 0.1, 0.1]),
            "Fase ideal": rng.randint(3, 8, n),
            "Fase": rng.randint(1, 6, n),
            "Ano nasc": rng.randint(2005, 2012, n),
            "Idade 22": rng.randint(10, 17, n),
            "Ano ingresso": rng.randint(2015, 2021, n),
            "INDE 22": rng.uniform(3, 9, n),
            "Cg": rng.uniform(3, 9, n),
            "Cf": rng.uniform(3, 9, n),
            "Ct": rng.uniform(3, 9, n),
            "Nº Av": rng.randint(1, 5, n),
            "IAA": rng.uniform(3, 9, n),
            "IEG": rng.uniform(3, 9, n),
            "IPS": rng.uniform(3, 9, n),
            "IDA": rng.uniform(3, 9, n),
            "IPV": rng.uniform(3, 9, n),
            "IAN": rng.uniform(3, 9, n),
            "Matem": rng.uniform(3, 9, n),
            "Portug": rng.uniform(3, 9, n),
            "Inglês": rng.uniform(3, 9, n),
            "Pedra 20": rng.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
            "Pedra 21": rng.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
            "Pedra 22": rng.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
            "Indicado": rng.choice(["Sim", "Não"], n),
            "Atingiu PV": rng.choice(["Sim", "Não"], n),
            "Rec Psicologia": rng.choice(["ok", "requer avaliação"], n),
        }
    )
    return df


# ------------------------------------------------------------------
# get_model
# ------------------------------------------------------------------


class TestGetModel:
    def test_returns_logistic_regression(self) -> None:
        from sklearn.linear_model import LogisticRegression

        model = get_model()
        assert isinstance(model, LogisticRegression)
        assert model.class_weight == "balanced"
        assert model.random_state == 42


# ------------------------------------------------------------------
# create_custom_scorers
# ------------------------------------------------------------------


class TestCreateCustomScorers:
    def test_returns_dict_with_expected_keys(self) -> None:
        scorers = create_custom_scorers()
        assert "f1_macro" in scorers
        assert "recall_macro" in scorers
        assert "accuracy" in scorers
        assert "recall_alto" in scorers


# ------------------------------------------------------------------
# save_model
# ------------------------------------------------------------------


class TestSaveModel:
    def test_creates_pkl_and_json(self, tmp_path: Path) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        metrics = {"accuracy": 0.9, "f1_macro": 0.85}
        features = ["feat_a", "feat_b"]
        training_info = {"training_data_shape": [100, 2], "test_data_shape": [20, 2]}

        path = save_model(pipe, metrics, features, training_info, tmp_path)

        assert path.exists()
        assert path.suffix == ".joblib"
        assert (tmp_path / "model_metadata.json").exists()

    def test_metadata_content(self, tmp_path: Path) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        metrics = {"accuracy": 0.9, "f1_macro": 0.85}
        features = ["feat_a", "feat_b"]
        training_info = {"training_data_shape": [100, 2], "test_data_shape": [20, 2]}

        save_model(pipe, metrics, features, training_info, tmp_path)

        with open(tmp_path / "model_metadata.json") as f:
            meta = json.load(f)

        assert meta["version"] == "1.0.0"
        assert meta["features"] == ["feat_a", "feat_b"]
        assert meta["metrics"]["test_accuracy"] == 0.9
        assert "trained_at" in meta
        assert meta["class_order"] == CLASS_ORDER

    def test_saved_model_is_loadable(self, tmp_path: Path) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        metrics = {}
        features = []
        training_info = {}

        path = save_model(pipe, metrics, features, training_info, tmp_path)

        loaded = joblib.load(path)
        assert isinstance(loaded, Pipeline)


# ------------------------------------------------------------------
# train_model (integration-level)
# ------------------------------------------------------------------


class TestTrainModel:
    def test_train_returns_pipeline_and_results(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        pipeline, results = train_model(
            sample_df, save=True, output_dir=tmp_path
        )
        assert pipeline is not None
        assert "cv_metrics" in results
        assert "test_metrics" in results
        assert results["model_name"] == "LogisticRegression"
        assert (tmp_path / "model.joblib").exists()
        assert (tmp_path / "model_metadata.json").exists()

    def test_train_without_save(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        pipeline, results = train_model(
            sample_df, save=False, output_dir=tmp_path
        )
        assert pipeline is not None
        assert not (tmp_path / "model.joblib").exists()
