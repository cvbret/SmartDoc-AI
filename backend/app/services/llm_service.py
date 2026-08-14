import os

from dotenv import load_dotenv
from openai import OpenAI
from app.utils.logger import logger

load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
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
