"""FastAPI app factory and router registration."""

from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Mortgage underwriting guideline analysis backend scaffold.",
    )
    app.include_router(health_router)
    return app
