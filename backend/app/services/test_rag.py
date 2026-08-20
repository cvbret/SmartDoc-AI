from app.services.rag_service import retrieve_context


question = "公司有什么福利？"


context = retrieve_context(
    question
)


print("检索结果:")
print(context)