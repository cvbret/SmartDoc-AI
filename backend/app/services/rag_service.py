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
        query_embedding
    )

    documents = result["documents"][0]


    context = "\n".join(
        documents
    )


    # 2. 构造Prompt

    prompt = f"""
你是一个企业知识助手。

请根据下面提供的资料回答问题。

如果资料中没有答案，请明确说明不知道。

资料:
{context}


问题:
{question}
"""
    
    messages = history.copy()


    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # 3. 调用LLM

    answer = chat_with_llm(
        messages
    )


    return answer


# 从 Chroma 中检索最相似的文本块
def retrieve_context(
    question: str
) -> str:

    query_embedding = generate_embedding(
        question
    )

    result = search_chunks(
        query_embedding
    )


    documents = result["documents"][0]


    return "\n".join(documents) #分隔符.join(列表)意思：使用指定的分隔符，把列表中的多个字符串连接起来。