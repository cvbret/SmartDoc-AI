"""Shared production prompt; importing this module has no service side effects."""


def build_rag_messages(
    question: str, history: list[dict], context: str,
) -> list[dict]:
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

    return messages
