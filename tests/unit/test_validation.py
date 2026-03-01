"""Tests for feature validation logic."""

import pytest

from app.validation import MissingFeaturesError, validate_features, validate_request_features


EXPECTED_FEATURES = [
    "numeric__Fase",
    "numeric__IAA",
    "numeric__IEG",
    "categorical__Gênero_Menino",
]


class TestValidateFeatures:
    def test_exact_features_passes(self) -> None:
        """All expected features present, no extras."""
        features = {"Fase": 1, "IAA": 7.5, "IEG": 8.0, "Gênero_Menino": 1}
        missing, extra = validate_features(features, EXPECTED_FEATURES)
        assert missing == []
        assert extra == []

    def test_missing_features_detected(self) -> None:
        """Some expected features are absent."""
        features = {"Fase": 1, "IAA": 7.5}
        missing, extra = validate_features(features, EXPECTED_FEATURES)
        assert "IEG" in missing
        assert "Gênero_Menino" in missing
        assert len(missing) == 2
        assert extra == []

    def test_extra_features_detected(self) -> None:
        """Unknown features present alongside all expected ones."""
        features = {
            "Fase": 1,
            "IAA": 7.5,
            "IEG": 8.0,
            "Gênero_Menino": 1,
            "UNKNOWN": 99,
        }
        missing, extra = validate_features(features, EXPECTED_FEATURES)
        assert missing == []
        assert "UNKNOWN" in extra

    def test_both_missing_and_extra(self) -> None:
        features = {"Fase": 1, "UNKNOWN": 99}
        missing, extra = validate_features(features, EXPECTED_FEATURES)
        assert len(missing) == 3  # IAA, IEG, Gênero_Menino
        assert "UNKNOWN" in extra

    def test_empty_features_all_missing(self) -> None:
        features: dict = {}
        missing, extra = validate_features(features, EXPECTED_FEATURES)
        assert len(missing) == 4
        assert extra == []

    def test_empty_expected_features_no_validation(self) -> None:
        """When expected features list is empty, validation is skipped."""
        features = {"Fase": 1, "IAA": 7.5}
        missing, extra = validate_features(features, [])
        assert missing == []
        assert extra == []

    def test_prefixed_feature_names_stripped(self) -> None:
        """Expected feature names have numeric__/categorical__ prefixes that
        should be stripped before matching against input keys."""
        features = {"Fase": 1, "IAA": 7.5, "IEG": 8.0, "Gênero_Menino": 1}
        expected = [
            "numeric__Fase",
            "numeric__IAA",
            "numeric__IEG",
            "categorical__Gênero_Menino",
        ]
        missing, extra = validate_features(features, expected)
        assert missing == []
        assert extra == []

    def test_feature_order_preserved(self) -> None:
        """Missing features should be returned in the order they appear
        in the expected list."""
        features: dict = {}
        expected = ["numeric__A", "numeric__B", "categorical__C"]
        missing, _ = validate_features(features, expected)
        assert missing == ["A", "B", "C"]


class TestValidateRequestFeatures:
    """Tests for the high-level validate_request_features function."""

    def test_raises_on_missing_features(self) -> None:
        metadata = {"input_features": ["A", "B", "C"]}
        with pytest.raises(MissingFeaturesError) as exc_info:
            validate_request_features({"A": 1}, metadata)
        assert "B" in exc_info.value.missing
        assert "C" in exc_info.value.missing

    def test_passes_when_all_present(self) -> None:
        metadata = {"input_features": ["A", "B"]}
        validate_request_features({"A": 1, "B": 2}, metadata)  # no exception

    def test_extra_features_tolerated(self) -> None:
        metadata = {"input_features": ["A"]}
        validate_request_features({"A": 1, "EXTRA": 2}, metadata)  # no exception

    def test_prefers_input_features_over_features(self) -> None:
        metadata = {
            "input_features": ["X", "Y"],
            "features": ["numeric__A", "numeric__B"],
        }
        # Should validate against input_features, not features
        validate_request_features({"X": 1, "Y": 2}, metadata)

    def test_falls_back_to_features_key(self) -> None:
        metadata = {"features": ["numeric__A", "numeric__B"]}
        with pytest.raises(MissingFeaturesError):
            validate_request_features({"A": 1}, metadata)

    def test_skips_when_no_metadata(self) -> None:
        validate_request_features({"A": 1}, {})  # no exception

    def test_error_contains_expected_list(self) -> None:
        metadata = {"input_features": ["A", "B", "C"]}
        with pytest.raises(MissingFeaturesError) as exc_info:
            validate_request_features({}, metadata)
        assert exc_info.value.expected == ["A", "B", "C"]


class TestMissingFeaturesError:
    """Tests for the MissingFeaturesError exception."""

    def test_attributes(self) -> None:
        err = MissingFeaturesError(missing=["A"], expected=["A", "B"])
        assert err.missing == ["A"]
        assert err.expected == ["A", "B"]

    def test_str_contains_missing(self) -> None:
        err = MissingFeaturesError(missing=["X", "Y"], expected=["X", "Y", "Z"])
        assert "X" in str(err)
        assert "Y" in str(err)
