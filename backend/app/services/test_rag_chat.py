from app.services.rag_service import rag_chat


question = "公司给员工提供哪些福利？"


answer = rag_chat(
    question
)


print(answer)