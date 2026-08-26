"""FastAPI application entry point."""

from app.core.logging import setup_logging


setup_logging()

from fastapi import FastAPI

from app.api import api_router
from app.exceptions.base import AppException
from app.exceptions.handlers import (
    app_exception_handler,
    unhandled_exception_handler
)



def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="SmartDoc AI",
        description="Enterprise AI document question-answering backend",
        version="0.1.0",
    )

    application.add_exception_handler(
        AppException,
        app_exception_handler
    )

    application.add_exception_handler(
        Exception,
        unhandled_exception_handler
    )

    application.include_router(api_router, prefix="/api")
    return application


app = create_app()
