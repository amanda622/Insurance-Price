"""Data access for predictions. One job: read/write ``Prediction`` rows.

Keeping SQL here means route handlers never build queries themselves.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from insurance_price.db.models import Prediction
from insurance_price.schemas import PredictionRequest, PredictionResponse


def save_prediction(
    session: Session,
    request: PredictionRequest,
    response: PredictionResponse,
) -> Prediction:
    """Persist a prediction request together with its result."""
    record = Prediction(
        age=request.age,
        sex=request.sex.value,
        bmi=request.bmi,
        children=request.children,
        smoker=request.smoker.value,
        region=request.region.value,
        predicted_charge=response.predicted_charge,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def list_predictions(session: Session, limit: int = 20) -> list[Prediction]:
    """Return the most recent predictions, newest first."""
    stmt = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    return list(session.scalars(stmt))
