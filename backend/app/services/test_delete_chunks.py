from app.services.vector_service import (
    delete_chunks_by_document_id,
    get_chunks_by_document_id
)


delete_chunks_by_document_id(
    7
)


result = get_chunks_by_document_id(
    7
)


print(result)