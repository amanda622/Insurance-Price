"""Tests for the feature pipeline."""

import pandas as pd

from insurance_price.features import FEATURE_COLUMNS, build_pipeline


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"age": 25, "sex": "male", "bmi": 22.0, "children": 0, "smoker": "no", "region": "northeast"},  # noqa: E501
            {"age": 52, "sex": "female", "bmi": 31.5, "children": 3, "smoker": "yes", "region": "southwest"},  # noqa: E501
            {"age": 40, "sex": "male", "bmi": 28.0, "children": 1, "smoker": "no", "region": "southeast"},  # noqa: E501
        ]
    )


def test_pipeline_has_expected_columns():
    assert FEATURE_COLUMNS == ["age", "bmi", "children", "sex", "smoker", "region"]


def test_pipeline_fits_and_predicts():
    X = _sample_frame()
    y = pd.Series([3000.0, 28000.0, 7000.0])

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)
    assert all(isinstance(float(p), float) for p in predictions)


def test_pipeline_handles_unknown_category():
    """OneHotEncoder is configured with handle_unknown='ignore'."""
    X = _sample_frame()
    y = pd.Series([3000.0, 28000.0, 7000.0])
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    unseen = X.copy()
    unseen.loc[0, "region"] = "atlantis"  # not in training data
    # Should not raise.
    assert len(pipeline.predict(unseen)) == len(unseen)
