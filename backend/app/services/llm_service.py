import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def chat_with_llm(message: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ]
    )

    return response.choices[0].message.content