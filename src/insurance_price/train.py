"""Training entry point: load data -> fit pipeline -> evaluate -> save artifacts.

Run with::

    python -m insurance_price.train

Produces ``models/model.joblib`` (the fitted pipeline) and ``models/metrics.json``
(evaluation metrics on a held-out test set).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from insurance_price.config import get_settings
from insurance_price.features import FEATURE_COLUMNS, TARGET_COLUMN, build_pipeline

RANDOM_STATE = 101
TEST_SIZE = 0.3


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and split it into features ``X`` and target ``y``."""
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE and R^2 on the held-out test set."""
    predictions = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "n_test": int(len(y_test)),
    }


def train(data_path: Path, model_path: Path) -> dict[str, float]:
    """Train the pipeline, save it, write metrics, and return the metrics."""
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path = model_path.with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = train(settings.data_path, settings.model_path)
    print(f"Saved model to {settings.model_path}")
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:,.4f}")


if __name__ == "__main__":
    main()
