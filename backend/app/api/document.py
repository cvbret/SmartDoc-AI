from fastapi import APIRouter, File, UploadFile, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.document_service import process_document
from app.schemas.response import ResponseModel
from app.services.document_crud import get_documents
from app.schemas.document import (
    DocumentResponse,
    DocumentListResponse
)

from app.services.document_service import (
    delete_document_service,
    update_document_service
)

from app.services.document_service import (
    get_document_detail
)

from app.exceptions.base import NotFoundException

router = APIRouter()


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    result = await process_document(
        db,
        file
    )


    return ResponseModel(
        data=result
    )


@router.put(
    "/documents/{document_id}"
)
async def update_document_api(
    document_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    result = await update_document_service(
        db,
        document_id,
        file
    )


    if not result:
        raise NotFoundException(
            "文档不存在"
        )


    return ResponseModel(
        data=result
    )

@router.get("/documents")
async def list_documents(
    db: Session = Depends(get_db),
    page: int = Query(
        default=1,
        ge=1
    ),
    page_size: int = Query(
        default=10,
        ge=1,
        le=100
    ),
    status: str | None = Query(
        default=None
    )
):

    documents, total = get_documents(
        db,
        page,
        page_size,
        status
    )


    total_pages = (
        total + page_size - 1
    ) // page_size


    return ResponseModel(
        data=DocumentListResponse(
            items=[
                DocumentResponse.model_validate(
                    document
                )
                for document in documents
            ],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages
        )
    )

@router.delete(
    "/documents/{document_id}"
)
async def delete_document_api(
    document_id: int,
    db: Session = Depends(get_db)
):

    result = delete_document_service(
        db,
        document_id
    )


    if not result:
        raise NotFoundException(
            "文档不存在"
        )


    return ResponseModel(
        data={
            "document_id": document_id,
            "message": "删除成功"
        }
    )

@router.get(
    "/documents/{document_id}"
)
async def get_document_detail_api(
    document_id: int,
    db: Session = Depends(get_db)
):

    document = get_document_detail(
        db,
        document_id
    )


    if not document:
        raise NotFoundException(
            "文档不存在"
        )


    return ResponseModel(
        data=DocumentResponse.model_validate(document)
    )
