"""Phase 1C: negative-prediction investigation for the current production model.

Fits the current production pipeline (``insurance_price.features.build_pipeline``)
in memory, reproducing ``insurance_price.train``'s training conditions exactly
(same dataset, same split, same random state), then checks whether it predicts
negative insurance charges over two separate grids:

- OBSERVED-RANGE grid: bounded by the real min/max values seen in the training
  dataset (interpolation region).
- LEGAL-RANGE grid: bounded by the Pydantic request schema (includes the
  legal-but-unobserved extrapolation region).

This script only READS production code. It never writes ``models/model.joblib``
and never calls ``insurance_price.train.train()``. It collects evidence only:
it does not choose a clamping policy, restrict the API, or recommend a fix.

Run from the repository root (with the project installed)::

    python -m experiments.investigate_negative_predictions
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from insurance_price.config import get_settings
from insurance_price.features import FEATURE_COLUMNS, build_pipeline
from insurance_price.schemas import Region, Sex, Smoker
from insurance_price.train import RANDOM_STATE, TEST_SIZE, load_dataset

ROUND_DECIMALS = 6
MAX_EXAMPLES = 20

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "experiments" / "results" / "negative_predictions.json"

# Schema-legal numeric boundaries, matching the Field constraints in
# insurance_price.schemas.PredictionRequest (age: ge=0, le=120 | bmi: gt=0,
# lt=100 | children: ge=0, le=20). bmi bounds are exclusive on both ends.
LEGAL_AGE_MIN, LEGAL_AGE_MAX = 0, 120
LEGAL_BMI_MIN, LEGAL_BMI_MAX = 0.0, 100.0
LEGAL_CHILDREN_MIN, LEGAL_CHILDREN_MAX = 0, 20

CATEGORICAL_VALUES = {
    "sex": [member.value for member in Sex],
    "smoker": [member.value for member in Smoker],
    "region": [member.value for member in Region],
}


def _linspace(low: float, high: float, count: int) -> list[float]:
    """Deterministic evenly spaced points from low to high, inclusive."""
    if count == 1:
        return [low]
    step = (high - low) / (count - 1)
    return [low + i * step for i in range(count)]


def _fit_current_production_model() -> tuple[object, pd.DataFrame, int]:
    """Fit build_pipeline(degree=2) using train.py's exact split settings.

    Reproduces the fitted pipeline currently shipped in models/model.joblib
    without reading or writing that file, so the investigation never touches
    a production artifact.
    """
    settings = get_settings()
    X, y = load_dataset(settings.data_path)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline(degree=2)
    pipeline.fit(X_train, y_train)
    return pipeline, X, len(X_train)


def _observed_numeric_bounds(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Real min/max for the numeric columns, read from the dataset."""
    return {
        "age": {"min": int(X["age"].min()), "max": int(X["age"].max())},
        "bmi": {"min": float(X["bmi"].min()), "max": float(X["bmi"].max())},
        "children": {"min": int(X["children"].min()), "max": int(X["children"].max())},
    }


def _build_grid_rows(
    ages: list[int], bmis: list[float], children_values: list[int]
) -> list[dict]:
    """Cartesian product of numeric grid points and all legal category values."""
    return [
        {
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex": sex,
            "smoker": smoker,
            "region": region,
        }
        for age, bmi, children in itertools.product(ages, bmis, children_values)
        for sex, smoker, region in itertools.product(
            CATEGORICAL_VALUES["sex"],
            CATEGORICAL_VALUES["smoker"],
            CATEGORICAL_VALUES["region"],
        )
    ]


def _bounds_of(rows: list[dict], key: str) -> dict[str, float] | None:
    if not rows:
        return None
    values = [row[key] for row in rows]
    return {"min": min(values), "max": max(values)}


def _investigate(pipeline, rows: list[dict], grid_definition: dict) -> dict:
    """Predict over the grid and summarize where negative predictions occur."""
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    predictions = pipeline.predict(frame)

    for row, prediction in zip(rows, predictions, strict=True):
        row["prediction"] = round(float(prediction), ROUND_DECIMALS)

    total_points = len(rows)
    negative_rows = [row for row in rows if row["prediction"] < 0]
    negative_count = len(negative_rows)

    negative_rows.sort(
        key=lambda row: (
            row["prediction"],
            row["age"],
            row["bmi"],
            row["children"],
            row["sex"],
            row["smoker"],
            row["region"],
        )
    )

    return {
        "grid_definition": grid_definition,
        "total_points": total_points,
        "negative_count": negative_count,
        "negative_rate": round(negative_count / total_points, ROUND_DECIMALS)
        if total_points
        else 0.0,
        "minimum_prediction": round(float(min(predictions)), ROUND_DECIMALS),
        "negative_numeric_bounds": {
            "age": _bounds_of(negative_rows, "age"),
            "bmi": _bounds_of(negative_rows, "bmi"),
            "children": _bounds_of(negative_rows, "children"),
        },
        "most_negative_examples": negative_rows[:MAX_EXAMPLES],
    }


def run() -> dict[str, object]:
    """Fit the current production model and run both grid investigations."""
    pipeline, X, training_row_count = _fit_current_production_model()
    observed_bounds = _observed_numeric_bounds(X)

    # Grid 1: observed range. Age/bmi sample 8 evenly spaced points across the
    # real dataset span; children uses every integer since the observed span
    # is small (a handful of distinct values).
    age_bounds = observed_bounds["age"]
    observed_ages = sorted(
        {round(v) for v in _linspace(age_bounds["min"], age_bounds["max"], 8)}
    )
    observed_bmis = sorted(
        {
            round(v, 2)
            for v in _linspace(observed_bounds["bmi"]["min"], observed_bounds["bmi"]["max"], 8)
        }
    )
    observed_children = list(
        range(observed_bounds["children"]["min"], observed_bounds["children"]["max"] + 1)
    )
    observed_grid_rows = _build_grid_rows(observed_ages, observed_bmis, observed_children)
    observed_grid_definition = {
        "age_points": observed_ages,
        "bmi_points": observed_bmis,
        "children_points": observed_children,
        "categorical_combinations": len(CATEGORICAL_VALUES["sex"])
        * len(CATEGORICAL_VALUES["smoker"])
        * len(CATEGORICAL_VALUES["region"]),
    }
    observed_investigation = _investigate(pipeline, observed_grid_rows, observed_grid_definition)

    # Grid 2: legal range. Coarse deterministic steps across the full
    # Pydantic-legal numeric range, including the legal-but-unobserved
    # extrapolation region. bmi endpoints stay strictly inside (0, 100).
    legal_ages = list(range(LEGAL_AGE_MIN, LEGAL_AGE_MAX + 1, 10))
    legal_bmis = [round(v, 2) for v in _linspace(1.0, 99.0, 11)]
    legal_children = list(range(LEGAL_CHILDREN_MIN, LEGAL_CHILDREN_MAX + 1, 4))
    legal_grid_rows = _build_grid_rows(legal_ages, legal_bmis, legal_children)
    legal_grid_definition = {
        "age_points": legal_ages,
        "bmi_points": legal_bmis,
        "children_points": legal_children,
        "categorical_combinations": len(CATEGORICAL_VALUES["sex"])
        * len(CATEGORICAL_VALUES["smoker"])
        * len(CATEGORICAL_VALUES["region"]),
    }
    legal_investigation = _investigate(pipeline, legal_grid_rows, legal_grid_definition)

    return {
        "experiment": {
            "name": "negative_prediction_investigation",
            "model_id": "current_production_pipeline",
            "matches_comparison_candidate": "B4",
            "pipeline_description": "insurance_price.features.build_pipeline(degree=2)",
            "dataset_row_count": int(len(X)),
            "training_row_count": training_row_count,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "round_decimals": ROUND_DECIMALS,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        },
        "observed_ranges": observed_bounds,
        "legal_ranges": {
            "age": {"min": LEGAL_AGE_MIN, "max": LEGAL_AGE_MAX},
            "bmi": {"min": LEGAL_BMI_MIN, "max": LEGAL_BMI_MAX, "bounds": "exclusive"},
            "children": {"min": LEGAL_CHILDREN_MIN, "max": LEGAL_CHILDREN_MAX},
            "categorical_values": CATEGORICAL_VALUES,
        },
        "observed_range_investigation": observed_investigation,
        "legal_range_investigation": legal_investigation,
    }


def main() -> None:
    artifact = run()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
