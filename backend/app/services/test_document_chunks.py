from app.services.vector_service import (
    get_chunks_by_document_id
)


result = get_chunks_by_document_id(
    9
)


print(result)
