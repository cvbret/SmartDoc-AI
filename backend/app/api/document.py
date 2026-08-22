from fastapi import APIRouter, File, UploadFile, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.document_service import process_document
from app.schemas.response import ResponseModel
from app.services.document_crud import get_documents
from app.schemas.document import DocumentResponse

router = APIRouter()


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    try:

        result = await process_document(
            db,
            file
        )


        return ResponseModel(
            data=result
        )


    except ValueError as e:

        return ResponseModel(
            code=400,
            message=str(e),
            data=None
        )


    except Exception:

        return ResponseModel(
            code=500,
            message="文档处理失败",
            data=None
        )

@router.get("/documents")
async def list_documents(
    db: Session = Depends(get_db)
):

    documents = get_documents(
        db
    )


    return ResponseModel(
    data=[
        DocumentResponse.model_validate(
            document
        )
        for document in documents
    ]
    )