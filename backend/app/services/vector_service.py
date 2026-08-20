from pathlib import Path
import chromadb
import uuid


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = BASE_DIR / "chroma"


# 初始化ChromaDB客户端
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)# 数据存储路径
)

collection = client.get_or_create_collection(
    name="smartdoc"
)


def save_chunks(
    chunks: list[dict]
):

    ids = []
    documents = []
    embeddings = []
    metadatas = []


    for index, chunk in enumerate(chunks):
        #遍历一个可迭代对象时，同时获取“元素的索引”和“元素本身”

        # 为每个chunk生成一个唯一的ID
        ids.append(
            str(uuid.uuid4())
        )

        documents.append(
            chunk["content"]
        )

        embeddings.append(
            chunk["embedding"]
        )

        metadatas.append(
            chunk["metadata"]
        )


    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

def search_chunks(
    query_embedding: list[float],
    top_k: int = 3 #找几个最相似的文本块
):
    print(
        "当前collection数量:",
        collection.count() #当前collection中的文档数量
    )

    #拿用户问题的 embedding 向量，和 Chroma 中保存的所有 chunk 向量进行相似度计算，然后按照相似程度排序，返回最相近的 top_k 个 chunk。
    result = collection.query(
        query_embeddings=[
            query_embedding
        ],               #外面还有一个 []因为 Chroma 支持一次查询多个问题
        n_results=top_k  #返回 top_k 个最相似的文本块
    )
    print("Chroma查询结果:")
    print(result)

    return result