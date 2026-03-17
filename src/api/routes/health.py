"""Health endpoints for service monitoring and local smoke tests."""

from fastapi import APIRouter

from src.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck() -> dict[str, str]:
    """Return a simple health payload."""
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }
