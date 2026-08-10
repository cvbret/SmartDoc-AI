from fastapi import APIRouter

from app.services.llm_service import chat_with_llm


router = APIRouter()


@router.post("/chat")
def chat(message: str):

    answer = chat_with_llm(message)

    return {
        "answer": answer
    }
