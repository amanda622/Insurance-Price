"""Database engine, session factory and declarative base.

One job: own the SQLAlchemy connection plumbing so the rest of the app just
asks for a session.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from insurance_price.config import get_settings

settings = get_settings()

# SQLite needs this flag when used across threads (as Uvicorn does).
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def init_db() -> None:
    """Create tables that don't exist yet."""
    # Import models so they register on Base.metadata before create_all.
    from insurance_price.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    """Yield a database session and ensure it is closed afterwards."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
