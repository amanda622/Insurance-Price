"""Tests for the prediction repository."""

from insurance_price.db import repository
from insurance_price.schemas import PredictionRequest, PredictionResponse


def test_save_and_list_prediction(db_session):
    request = PredictionRequest(
        age=30, sex="male", bmi=24.0, children=0, smoker="no", region="northwest"
    )
    response = PredictionResponse(predicted_charge=4200.50)

    saved = repository.save_prediction(db_session, request, response)
    assert saved.id is not None
    assert saved.predicted_charge == 4200.50
    assert saved.smoker == "no"

    recent = repository.list_predictions(db_session, limit=5)
    assert any(r.id == saved.id for r in recent)


def test_list_respects_limit(db_session):
    request = PredictionRequest(
        age=30, sex="male", bmi=24.0, children=0, smoker="no", region="northwest"
    )
    response = PredictionResponse(predicted_charge=1000.0)
    for _ in range(3):
        repository.save_prediction(db_session, request, response)

    assert len(repository.list_predictions(db_session, limit=2)) == 2
