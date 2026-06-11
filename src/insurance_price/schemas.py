"""Pydantic request/response schemas.

One job: define the shape and validation rules of data crossing the API
boundary. Invalid input is rejected here (FastAPI returns 422 automatically),
so handlers and the model never see malformed data.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Sex(StrEnum):
    male = "male"
    female = "female"


class Smoker(StrEnum):
    yes = "yes"
    no = "no"


class Region(StrEnum):
    northeast = "northeast"
    northwest = "northwest"
    southeast = "southeast"
    southwest = "southwest"


class PredictionRequest(BaseModel):
    """A single insurance quote request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35,
                "sex": "female",
                "bmi": 27.5,
                "children": 2,
                "smoker": "no",
                "region": "southwest",
            }
        }
    )

    age: int = Field(ge=0, le=120, description="Age in years.")
    sex: Sex
    bmi: float = Field(gt=0, lt=100, description="Body mass index.")
    children: int = Field(ge=0, le=20, description="Number of dependents.")
    smoker: Smoker
    region: Region


class PredictionResponse(BaseModel):
    """The predicted insurance charge."""

    predicted_charge: float = Field(description="Predicted annual charge in USD.")
    currency: str = "USD"
