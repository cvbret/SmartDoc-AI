from openai import OpenAI
from app.utils.logger import logger
from app.core.config import settings

client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url
)


def chat_with_llm(messages: list[dict[str, str]]) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        return response.choices[0].message.content

    except Exception:
        logger.exception("LLM调用失败")
        raise
