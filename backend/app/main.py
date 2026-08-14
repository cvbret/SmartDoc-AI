"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api import api_router

from app.api import chat

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="SmartDoc AI",
        description="Enterprise AI document question-answering backend",
        version="0.1.0",
    )
    application.include_router(api_router, prefix="/api")
    return application


app = create_app()

app.include_router(
    chat.router,
    prefix="/api"
)#将chat的router注册到app中