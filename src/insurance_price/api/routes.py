"""HTTP routes. Handlers stay thin: validate -> predict -> log -> respond."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from insurance_price.api.deps import get_model_service, get_session
from insurance_price.db import repository
from insurance_price.model import ModelService
from insurance_price.schemas import (
    PredictionRequest,
    PredictionResponse,
    Region,
    Sex,
    Smoker,
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()

# Reusable dependency aliases (modern FastAPI style — keeps signatures readable).
ModelDep = Annotated[ModelService, Depends(get_model_service)]
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe for containers / load balancers."""
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    request: PredictionRequest,
    model: ModelDep,
    session: SessionDep,
) -> PredictionResponse:
    """Predict an insurance charge from a JSON body and log the request."""
    response = model.predict(request)
    repository.save_prediction(session, request, response)
    return response


@router.get("/", response_class=HTMLResponse, tags=["web"])
def home(request: Request) -> HTMLResponse:
    """Render the prediction form."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {"sexes": list(Sex), "smokers": list(Smoker), "regions": list(Region)},
    )


@router.post("/web/predict", response_class=HTMLResponse, tags=["web"])
def web_predict(
    request: Request,
    model: ModelDep,
    session: SessionDep,
    age: Annotated[int, Form()],
    sex: Annotated[Sex, Form()],
    bmi: Annotated[float, Form()],
    children: Annotated[int, Form()],
    smoker: Annotated[Smoker, Form()],
    region: Annotated[Region, Form()],
) -> HTMLResponse:
    """Handle the HTML form submission and render the result page."""
    prediction_request = PredictionRequest(
        age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region
    )
    response = model.predict(prediction_request)
    repository.save_prediction(session, prediction_request, response)
    return templates.TemplateResponse(
        request,
        "result.html",
        {"prediction": f"${response.predicted_charge:,.2f}"},
    )


@router.get("/predictions", tags=["prediction"])
def recent_predictions(
    session: SessionDep,
    limit: int = 20,
) -> list[dict]:
    """Return the most recent logged predictions."""
    records = repository.list_predictions(session, limit=limit)
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "age": r.age,
            "sex": r.sex,
            "bmi": r.bmi,
            "children": r.children,
            "smoker": r.smoker,
            "region": r.region,
            "predicted_charge": r.predicted_charge,
        }
        for r in records
    ]
