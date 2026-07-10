"""Phase 1B model comparison experiment (evidence collection only).

Compares five candidates (B1-B5) under a single, shared ``RepeatedKFold`` split
and writes mean / standard-deviation / per-fold RMSE, MAE and R^2 scores to
``experiments/results/results.json``.

This script only READS production code (``insurance_price.*``). It never modifies
the production pipeline and never writes the production model artifact. It does
not choose a "winner": Phase 2 interprets the evidence separately.

Run from the repository root (with the project installed)::

    python -m experiments.compare_models
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from insurance_price.config import get_settings
from insurance_price.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_pipeline,
)
from insurance_price.train import load_dataset

# Frozen experiment configuration (approved design, Phase 1B).
SEED = 101
N_SPLITS = 5
N_REPEATS = 5
STD_DDOF = 1  # sample standard deviation across fold scores

# The random forest reference runs with n_jobs=-1 (frozen by the design). Its
# parallel prediction accumulation reorders floating-point additions between
# runs, which perturbs the last few ULPs. Reported scores are rounded so the
# artifact is byte-identical across runs; the estimators themselves are untouched
# and 6 decimals is far finer than these scores meaningfully carry.
ROUND_DECIMALS = 6

# Scikit-learn scorers. RMSE and MAE scorers are negative (higher-is-better
# convention), so their per-fold scores are sign-flipped back to positive below.
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "experiments" / "results" / "results.json"


def _onehot() -> OneHotEncoder:
    """One-hot encoder matching the production encoding (no dropped categories)."""
    return OneHotEncoder(handle_unknown="ignore")


def _b1_dummy() -> DummyRegressor:
    """B1 - predict the training-set mean (no-information floor)."""
    return DummyRegressor(strategy="mean")


def _b2_main_effects_linear() -> Pipeline:
    """B2 - one-hot categoricals + passthrough numerics, no polynomial step."""
    preprocess = ColumnTransformer(
        transformers=[("categorical", _onehot(), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def _b3_numeric_polynomial() -> Pipeline:
    """B3 - polynomial expansion on numerics only; categoricals as main effects."""
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", _onehot(), CATEGORICAL_FEATURES),
            (
                "numeric_polynomial",
                PolynomialFeatures(degree=2, include_bias=False),
                NUMERIC_FEATURES,
            ),
        ],
    )
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def _b4_production_pipeline() -> Pipeline:
    """B4 - the real production pipeline, reused (not reimplemented)."""
    return build_pipeline(degree=2)


def _b5_tree_reference() -> Pipeline:
    """B5 - fixed random forest as a nonlinear reference (not tuned)."""
    preprocess = ColumnTransformer(
        transformers=[("categorical", _onehot(), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("model", model)])


# Each candidate: stable id, description, and a factory that builds a fresh
# estimator. Descriptions are hard-coded (not object reprs) so the artifact stays
# stable across scikit-learn versions.
CANDIDATES: list[dict[str, object]] = [
    {
        "id": "B1",
        "name": "Dummy mean baseline",
        "purpose": "Define the no-information floor (predicts the training-set mean).",
        "estimator": "DummyRegressor(strategy='mean')",
        "fixed_parameters": {"strategy": "mean"},
        "build": _b1_dummy,
    },
    {
        "id": "B2",
        "name": "Main-effects linear model",
        "purpose": "Isolate the value added by the current polynomial step.",
        "estimator": "OneHotEncoder + passthrough -> LinearRegression",
        "fixed_parameters": {
            "onehot_handle_unknown": "ignore",
            "polynomial": False,
        },
        "build": _b2_main_effects_linear,
    },
    {
        "id": "B3",
        "name": "Numeric-only polynomial model",
        "purpose": (
            "Measure numeric curvature and numeric interactions without "
            "category interactions."
        ),
        "estimator": (
            "OneHotEncoder + PolynomialFeatures(numeric only) -> LinearRegression"
        ),
        "fixed_parameters": {
            "onehot_handle_unknown": "ignore",
            "polynomial_degree": 2,
            "polynomial_include_bias": False,
            "polynomial_scope": "numeric_features_only",
        },
        "build": _b3_numeric_polynomial,
    },
    {
        "id": "B4",
        "name": "Current production pipeline",
        "purpose": "Test the real current production design (reused build_pipeline).",
        "estimator": "insurance_price.features.build_pipeline(degree=2)",
        "fixed_parameters": {"polynomial_degree": 2},
        "build": _b4_production_pipeline,
    },
    {
        "id": "B5",
        "name": "Random forest reference",
        "purpose": "Provide a fixed nonlinear reference (not a tuned competitor).",
        "estimator": "OneHotEncoder + passthrough -> RandomForestRegressor",
        "fixed_parameters": {
            "n_estimators": 300,
            "random_state": SEED,
            "n_jobs": -1,
        },
        "build": _b5_tree_reference,
    },
]


def _summarize(scores: np.ndarray) -> dict[str, object]:
    """Return mean, standard deviation and per-fold scores for one metric."""
    return {
        "mean": round(float(np.mean(scores)), ROUND_DECIMALS),
        "std": round(float(np.std(scores, ddof=STD_DDOF)), ROUND_DECIMALS),
        "per_fold": [round(float(score), ROUND_DECIMALS) for score in scores],
    }


def _evaluate(estimator: BaseEstimator, X, y, folds: list) -> dict[str, object]:
    """Cross-validate one candidate over the shared folds and collect metrics."""
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=folds,
        scoring=SCORING,
        error_score="raise",
    )
    return {
        "rmse": _summarize(-scores["test_rmse"]),
        "mae": _summarize(-scores["test_mae"]),
        "r2": _summarize(scores["test_r2"]),
    }


def run() -> dict[str, object]:
    """Run the comparison and return the artifact dictionary."""
    settings = get_settings()
    X, y = load_dataset(settings.data_path)

    # Create the split indices exactly once and reuse the same list for every
    # candidate, so all candidates are evaluated on identical folds.
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(cv.split(X))

    candidate_results = []
    for spec in CANDIDATES:
        build: Callable[[], BaseEstimator] = spec["build"]  # type: ignore[assignment]
        metrics = _evaluate(build(), X, y, folds)
        candidate_results.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "purpose": spec["purpose"],
                "estimator": spec["estimator"],
                "fixed_parameters": spec["fixed_parameters"],
                "metrics": metrics,
            }
        )

    return {
        "experiment": {
            "name": "model_comparison_b1_b5",
            "seed": SEED,
            "cv_type": "RepeatedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "n_folds_total": len(folds),
            "std_ddof": STD_DDOF,
            "rounding_decimals": ROUND_DECIMALS,
            "primary_metric": "rmse",
            "secondary_metrics": ["mae", "r2"],
            "dataset_row_count": int(len(X)),
            "feature_columns": list(FEATURE_COLUMNS),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "target_column": TARGET_COLUMN,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        },
        "candidates": candidate_results,
    }


def main() -> None:
    artifact = run()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
