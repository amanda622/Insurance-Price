"""Application configuration loaded from environment variables / ``.env``.

One job: expose a typed ``Settings`` object so no other module reads os.environ
or hard-codes paths.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = three levels up from this file (src/insurance_price/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_path: Path = PROJECT_ROOT / "models" / "model.joblib"
    data_path: Path = PROJECT_ROOT / "data" / "insurance.csv"
    database_url: str = "sqlite:///./predictions.db"


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
