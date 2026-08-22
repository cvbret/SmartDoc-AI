from app.db.database import SessionLocal
from app.services.document_crud import (
    create_document,
    update_document_status
)


db = SessionLocal()


document = create_document(
    db,
    "test2.txt",
    "text/plain",
    2048
)


print(
    "创建:",
    document.id,
    document.status
)


document = update_document_status(
    db,
    document.id,
    "completed"
)


print(
    "更新:",
    document.id,
    document.status
)


db.close()