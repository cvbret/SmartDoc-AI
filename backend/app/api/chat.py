from fastapi import APIRouter

from app.services.llm_service import chat_with_llm
from app.schemas.chat import ChatRequest
from app.schemas.response import ResponseModel

router = APIRouter()


@router.post("/chat")
def chat(request: ChatRequest):

    answer = chat_with_llm(
        request.message
        )

    return ResponseModel(
    data={
        "answer": answer
    }
    )
