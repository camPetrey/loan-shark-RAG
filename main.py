"""Application entrypoint for the Loan Shark FastAPI service."""

from src.api.app import create_app

app = create_app()
