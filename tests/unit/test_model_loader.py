import json
from pathlib import Path


from app.model_loader import load_metadata, load_model


class TestLoadModel:
    def test_load_valid_joblib_model(self, tmp_model_path: Path) -> None:
        model = load_model(tmp_model_path)
        assert model is not None

    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        model = load_model(tmp_path / "nonexistent.pkl")
        assert model is None

    def test_load_corrupted_file_returns_none(self, tmp_path: Path) -> None:
        corrupted = tmp_path / "corrupted.pkl"
        corrupted.write_text("this is not a valid pickle")
        model = load_model(corrupted)
        assert model is None

    def test_load_valid_pickle_model(self, tmp_path: Path) -> None:
        """Ensure fallback to pickle works when joblib fails."""
        import pickle

        dummy = {"type": "dummy_model"}
        pkl_path = tmp_path / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(dummy, f)

        model = load_model(pkl_path)
        assert model is not None
        assert model["type"] == "dummy_model"


class TestLoadMetadata:
    def test_load_valid_metadata(self, tmp_path: Path) -> None:
        meta_path = tmp_path / "metadata.json"
        meta = {
            "version": "2.0.0",
            "metrics": {"f1": 0.92},
            "features": ["a", "b"],
        }
        meta_path.write_text(json.dumps(meta))

        result = load_metadata(meta_path)
        assert result["version"] == "2.0.0"
        assert result["metrics"] == {"f1": 0.92}
        assert result["features"] == ["a", "b"]

    def test_load_missing_metadata_returns_defaults(self, tmp_path: Path) -> None:
        result = load_metadata(tmp_path / "nonexistent.json")
        assert result["version"] == "unknown"
        assert result["metrics"] == {}
        assert result["features"] == []

    def test_load_corrupted_metadata_returns_defaults(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("NOT JSON {{{")

        result = load_metadata(bad)
        assert result["version"] == "unknown"

    def test_partial_metadata_merges_with_defaults(self, tmp_path: Path) -> None:
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"version": "3.0.0"}))

        result = load_metadata(meta_path)
        assert result["version"] == "3.0.0"
        assert result["metrics"] == {}
        assert result["features"] == []
