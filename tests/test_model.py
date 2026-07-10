"""Tests for the ModelService."""

from pathlib import Path

import pytest

from insurance_price.config import get_settings
from insurance_price.model import ModelNotFoundError, ModelService
from insurance_price.schemas import PredictionRequest


def test_predict_returns_positive_charge():
    service = ModelService(get_settings().model_path)
    request = PredictionRequest(
        age=35, sex="female", bmi=27.5, children=2, smoker="no", region="southwest"
    )
    response = service.predict(request)
    assert response.predicted_charge > 0
    assert response.currency == "USD"


def test_smoker_costs_more_than_non_smoker():
    service = ModelService(get_settings().model_path)
    base = dict(age=45, sex="male", bmi=30.0, children=1, region="southeast")
    smoker = service.predict(PredictionRequest(smoker="yes", **base))
    non_smoker = service.predict(PredictionRequest(smoker="no", **base))
    assert smoker.predicted_charge > non_smoker.predicted_charge


def test_missing_model_raises():
    with pytest.raises(ModelNotFoundError):
        ModelService(Path("/nonexistent/model.joblib"))


def test_predict_never_returns_negative_charge():
    """Regression case from experiments/results/negative_predictions.json.

    This exact input is inside the dataset's observed numeric range and is
    a plausible real-world quote, yet the unguarded model predicted
    approximately -3414.52.
    """
    service = ModelService(get_settings().model_path)
    request = PredictionRequest(
        age=18, sex="male", bmi=53.13, children=0, smoker="no", region="southeast"
    )
    response = service.predict(request)
    assert response.predicted_charge >= 0
