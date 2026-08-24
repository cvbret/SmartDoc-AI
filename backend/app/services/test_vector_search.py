from app.services.vector_service import collection


result = collection.get(
   where={
        "document_id":8
    }
)


for metadata in result["metadatas"]:
    print(metadata)