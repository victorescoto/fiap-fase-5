"""Feature validation utilities.

Validates user-supplied feature dictionaries against the expected
feature names stored in the model metadata.  Metadata feature names
may carry ``numeric__`` or ``categorical__`` prefixes from the
sklearn ``ColumnTransformer``; the validation logic strips these
prefixes transparently so callers can use *raw* column names.
"""

from __future__ import annotations

import re

_PREFIX_RE = re.compile(r"^(?:numeric|categorical)__")


def _strip_prefix(name: str) -> str:
    """Remove the ``numeric__`` or ``categorical__`` prefix."""
    return _PREFIX_RE.sub("", name)


def validate_features(
    features: dict,
    expected_features: list[str],
) -> tuple[list[str], list[str]]:
    """Compare *features* keys against *expected_features*.

    Args:
        features: Dictionary sent by the client (key → value).
        expected_features: Feature names from model metadata.  May
            contain ``numeric__`` / ``categorical__`` prefixes which
            are stripped before comparison.

    Returns:
        A ``(missing, extra)`` tuple where *missing* lists expected
        feature names absent from the input and *extra* lists input
        keys that are not in the expected set.  Both lists use the
        *stripped* (raw) names.  If *expected_features* is empty the
        validation is skipped and both lists are empty.
    """
    if not expected_features:
        return [], []

    stripped_expected = [_strip_prefix(f) for f in expected_features]
    expected_set = set(stripped_expected)
    provided_set = set(features.keys())

    missing = [f for f in stripped_expected if f not in provided_set]
    extra = sorted(provided_set - expected_set)

    return missing, extra
