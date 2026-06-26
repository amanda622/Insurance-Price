"""FastAPI application factory.

One job: assemble the app — mount routes and static files, register error
handlers, and create database tables on startup.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from insurance_price.api.routes import router
from insurance_price.db.base import init_db
from insurance_price.model import ModelNotFoundError

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables before the app starts serving."""
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Insurance Price Prediction API",
        description="Predict health insurance charges from patient attributes.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


app = create_app()
