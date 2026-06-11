"""Model serving: load the trained pipeline once and turn requests into responses.

``ModelService`` is the only object that touches the joblib artifact. It knows
nothing about HTTP or databases.
"""

from pathlib import Path

import joblib
import pandas as pd

from insurance_price.features import FEATURE_COLUMNS
from insurance_price.schemas import PredictionRequest, PredictionResponse


class ModelNotFoundError(RuntimeError):
    """Raised when the model artifact is missing — run training first."""


class ModelService:
    """Loads a fitted pipeline and produces predictions."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Run `python -m insurance_price.train` first."
            )
        self._pipeline = joblib.load(model_path)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Predict the insurance charge for a single request."""
        row = pd.DataFrame([request.model_dump(mode="json")], columns=FEATURE_COLUMNS)
        charge = float(self._pipeline.predict(row)[0])
        return PredictionResponse(predicted_charge=round(charge, 2))
