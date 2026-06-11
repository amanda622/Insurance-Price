"""ORM models. One job: describe database tables."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from insurance_price.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Prediction(Base):
    """An audit record of one prediction request and its result."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Inputs
    age: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(10))
    bmi: Mapped[float] = mapped_column(Float)
    children: Mapped[int] = mapped_column(Integer)
    smoker: Mapped[str] = mapped_column(String(5))
    region: Mapped[str] = mapped_column(String(20))

    # Output
    predicted_charge: Mapped[float] = mapped_column(Float)
