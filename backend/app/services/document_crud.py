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


def update_document(
    db: Session,
    document_id: int,
    filename: str,
    file_type: str | None,
    file_size: int | None
):

    document = (
        db.query(Document)
        .filter(
            Document.id == document_id
        )
        .first()
    )


    if not document:
        return None


    document.filename = filename
    document.file_type = file_type
    document.file_size = file_size
    document.status = "processing"

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
    db: Session,
    page: int = 1,
    page_size: int = 10,
    status: str | None = None
):

    query = db.query(Document)


    if status is not None:
        query = query.filter(
            Document.status == status
        )


    total = query.count()

    offset = (page - 1) * page_size

    documents = (
        query
        .order_by(
            Document.id.desc()
        )
        .offset(offset)
        .limit(page_size)
        .all()
    )


    return documents, total

def delete_document(
    db: Session,
    document_id: int
):

    document = (
        db.query(Document)
        .filter(
            Document.id == document_id
        )
        .first()
    )


    if not document:
        return False


    db.delete(document)

    db.commit()

    return True

def get_document_by_id(
    db: Session,
    document_id: int
):

    document = (
        db.query(Document)
        .filter(
            Document.id == document_id
        )
        .first()
    )

    return document
