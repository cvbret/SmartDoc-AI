"""Health check API endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict[str, str]:#暂时不知道health什么时候注册的
    """Report whether the backend service is running."""
    return {
        "status": "ok",
        "message": "SmartDoc AI backend running",
    }
