from app.services.embedding_service import embed_chunks


chunks = [
    {
        "content":"员工每年享受带薪年假",
        "metadata":{
            "filename":"test.pdf",
            "chunk_id":0
        }
    },
    {
        "content":"公司提供五险一金",
        "metadata":{
            "filename":"test.pdf",
            "chunk_id":1
        }
    }
]


result = embed_chunks(chunks)


print(result[0].keys())

print(
    len(result[0]["embedding"])
)