from fastapi import APIRouter, File, UploadFile

from app.schemas.response import ResponseModel
from app.services.document_service import parse_document
from app.services.chunk_service import split_text
from app.services.embedding_service import embed_chunks
from app.services.vector_service import save_chunks

router = APIRouter()


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)#表示这个参数接收上传文件。
    #参数名：数据类型 = 默认值  ...表示一个真实的对象,表示必须提供一个值
):
    try:
        text = await parse_document(file)
        chunks = split_text(
            text,
            file.filename
        )
        chunks = split_text(
    text,
    file.filename
)


        embedded_chunks = embed_chunks(
            chunks
        )

        save_chunks(
            embedded_chunks
        )

        return ResponseModel(
            data={
                "filename": file.filename,
                "content_type": file.content_type,
                "text_length": len(text),
                "chunk_count": len(chunks)
            }
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
            message="文档解析失败",
            data=None
        )