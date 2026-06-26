"""Feature engineering: defines the columns and builds the model pipeline.

This module has a single job — describe how raw insurance inputs are turned
into model features and return one scikit-learn ``Pipeline`` that bundles
preprocessing and the estimator together. Bundling them means the fitted
pipeline only ever ``.transform()``s at inference time, which avoids the
classic "re-fit the transformer on a single row" bug.
"""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

# Raw input columns expected from the dataset and the API.
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "charges"


def build_pipeline(degree: int = 2) -> Pipeline:
    """Build an unfitted pipeline: encode -> polynomial expansion -> linear model.

    Args:
        degree: Degree of the polynomial feature expansion.

    Returns:
        An unfitted scikit-learn ``Pipeline``.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # keep numeric features as-is
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )
