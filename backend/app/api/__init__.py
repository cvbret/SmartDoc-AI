"""API router registration."""

from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.document import router as document_router
from fastapi import APIRouter

api_router = APIRouter()


api_router.include_router(
    health_router
)

api_router.include_router(
    chat_router
)

api_router.include_router(
    document_router
)

