from app.services.vector_service import search_chunks


query_vector = [
    0.1,
    0.2,
    0.3
]


result = search_chunks(
    query_vector
)


print(result)