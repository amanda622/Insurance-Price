"""Shared test fixtures.

Tests run against isolated temp paths so they never touch the real database or
the committed model artifact. Environment variables are set *before* importing
the app so configuration picks them up.
"""

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="insurance-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["MODEL_PATH"] = str(_TMP / "model.joblib")

from insurance_price.config import get_settings  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Train a model into the temp path once for the whole test session."""
    from insurance_price.train import train

    settings = get_settings()
    train(settings.data_path, settings.model_path)
    yield


@pytest.fixture
def db_session():
    """A database session backed by the temp test database."""
    from insurance_price.db.base import Base, SessionLocal, engine

    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """A TestClient with a freshly built app and reset model cache."""
    from fastapi.testclient import TestClient

    from insurance_price.api.app import create_app
    from insurance_price.api.deps import get_model_service

    get_model_service.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
