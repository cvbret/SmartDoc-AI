from sqlalchemy.orm import Session

from app.models.document import Document



def create_document(
    db: Session,
    filename: str,
    file_type: str | None,
    file_size: int | None
):

    document = Document(
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        status="processing"
    )


    db.add(document)

    db.commit()

    db.refresh(document)


    return document

def update_document_status(
    db: Session,
    document_id: int,
    status: str
):

    document = (
        db.query(Document)
        .filter(
            Document.id == document_id
        )
        .first()
    )


    if document:

        document.status = status

        db.commit()

        db.refresh(document)


    return document

def get_documents(
    db: Session
):

    return (
        db.query(Document)
        .all()
    )