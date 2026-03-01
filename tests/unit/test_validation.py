"""Tests for feature validation logic."""

import pytest

from app.validation import validate_features


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
