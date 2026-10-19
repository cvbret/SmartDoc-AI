from app.services.prompt_service import build_rag_messages
from app.core.config import settings
from app.services.reranker_service import rerank_candidates
from app.services.retrieval_service import (
    normalize_candidates, select_candidates, build_context,
)
from app.services.embedding_service import generate_embedding
from app.services.vector_service import search_chunks
from app.services.llm_service import chat_with_llm

def rag_chat(
    question: str,
    history: list[dict]
) -> str:

    # 1. 检索相关文档
    query_embedding = generate_embedding(
        question
    )

    result = search_chunks(
        query_embedding, top_k=settings.retrieval_k
    )

    candidates = normalize_candidates(result)
    reranked = rerank_candidates(question, candidates)
    selected = select_candidates(reranked, settings.rerank_top_n)
    context = build_context(selected)


    # 2. 构造Prompt

    messages = build_rag_messages(question, history, context)

    # 3. 调用LLM

    answer = chat_with_llm(
        messages
    )


    return answer
