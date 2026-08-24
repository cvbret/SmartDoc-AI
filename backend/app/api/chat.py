from fastapi import APIRouter

from app.services.rag_service import rag_chat
from app.schemas.chat import ChatRequest
from app.schemas.response import ResponseModel
from app.services.conversation_service import (
    add_message,
    get_history,
)
from app.services.llm_service import chat_with_llm


router = APIRouter()#创建一个APIRouter对象，名字叫router


@router.post("/chat")
def chat(request: ChatRequest):

    add_message(
        request.session_id,
        "user",
        request.message
    )

    history = get_history(
        request.session_id
    )

    answer = rag_chat(
        request.message,
        history
    )

    add_message(
        request.session_id,
        "assistant",
        answer
    )

    return ResponseModel(
        data={
            "answer": answer
        }
    )
