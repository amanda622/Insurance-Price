"""FastAPI dependency providers.

One job: hand route handlers the collaborators they need (model service, db
session) without those handlers constructing anything themselves.
"""

from functools import lru_cache

from insurance_price.config import get_settings
from insurance_price.db.base import get_session  # re-exported for routes
from insurance_price.model import ModelService

__all__ = ["get_model_service", "get_session"]


@lru_cache
def get_model_service() -> ModelService:
    """Return a process-wide ModelService (the artifact is loaded once)."""
    return ModelService(get_settings().model_path)
