from app.services.vector_service import save_chunks


chunks = [
    {
        "content": "员工每年享受带薪年假",
        "embedding": [
            0.1,
            0.2,
            0.3
        ],
        "metadata": {
            "filename": "test.pdf",
            "chunk_id": 0
        }
    }
]


save_chunks(chunks)


print("保存成功")