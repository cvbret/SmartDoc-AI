from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"


model = SentenceTransformer(
    MODEL_NAME
)


def generate_embedding(
    text: str
) -> list[float]:
    
    # 生成嵌入向量
    vector = model.encode(
        text
    )

    return vector.tolist() # 转换为列表

def embed_chunks(
    chunks: list[dict]
) -> list[dict]:

    result = []

    for chunk in chunks:

        embedding = generate_embedding(
            chunk["content"]
        )

        result.append(
            {
                "content": chunk["content"],
                "embedding": embedding,
                "metadata": chunk["metadata"]
            }
        )

    return result