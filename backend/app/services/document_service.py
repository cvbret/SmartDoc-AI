import logging

from sqlalchemy.orm import Session

from app.services.document_crud import (
    create_document,
    update_document,
    update_document_status,
    get_document_by_id
)

from app.services.chunk_service import split_text
from app.services.embedding_service import embed_chunks
from app.services.vector_service import save_chunks

from io import BytesIO

from fastapi import UploadFile
from pypdf import PdfReader

from app.services.vector_service import (
    delete_chunks_by_document_id
)


from app.services.document_crud import (
    delete_document
)

from app.models.document import Document
from app.exceptions.base import BadRequestException

logger = logging.getLogger(__name__)


SUPPORTED_CONTENT_TYPES = {
    "text/plain",
    "application/pdf",
}


def _get_upload_file_size(
    file: UploadFile
) -> int:

    file.file.seek(
        0,
        2
    )

    file_size = file.file.tell()

    file.file.seek(
        0
    )

    return file_size


async def parse_document(file: UploadFile) -> str:
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise BadRequestException(
            "暂不支持该文件类型"
        )

    content = await file.read() # 把上传文件的原始字节读入 Python 内存。

    try:

        if file.content_type == "text/plain":
            return _parse_txt(content)

        if file.content_type == "application/pdf":
            return _parse_pdf(content)

    except ValueError as e:

        raise BadRequestException(
            str(e)
        ) from e

    raise BadRequestException(
        "无法识别文件类型"
    )

async def process_document(
    db: Session,
    file: UploadFile
):

    document = None

    try:

        file_size = _get_upload_file_size(
            file
        )


        # 1. 创建数据库记录
        document = create_document(
            db,
            file.filename,
            file.content_type,
            file_size
        )

        logger.info(
            "Processing document document_id=%s filename=%s content_type=%s size_bytes=%s",
            document.id,
            file.filename,
            file.content_type,
            file_size,
        )

        prepared_data = await _prepare_document_data(
            file,
            document.id
        )


        save_chunks(
            prepared_data["embedded_chunks"]
        )


        update_document_status(
            db,
            document.id,
            "completed"
        )

        logger.info(
            "Document processing completed document_id=%s chunk_count=%s",
            document.id,
            len(prepared_data["chunks"]),
        )


        return {
            "filename": file.filename,
            "text_length": len(prepared_data["text"]),
            "chunk_count": len(prepared_data["chunks"]),
            "document_id": document.id
        }


    except Exception:

        if document:

            update_document_status(
                db,
                document.id,
                "failed"
            )

            logger.warning(
                "Document processing failed document_id=%s",
                document.id,
            )

        raise


async def _prepare_document_data(
    file: UploadFile,
    document_id: int
):

    try:

        # 1. 解析文件
        text = await parse_document(file)


        # 2. chunk切分
        chunks = split_text(
            text,
            file.filename,
            document_id
        )


        # 3. embedding
        embedded_chunks = embed_chunks(
            chunks
        )

        return {
            "text": text,
            "chunks": chunks,
            "embedded_chunks": embedded_chunks
        }


    except ValueError as e:

        raise BadRequestException(
            str(e)
        ) from e

def _parse_txt(content: bytes) -> str:
    return content.decode(
        "utf-8"
    )


def _parse_pdf(content: bytes) -> str:
    reader = PdfReader(
        BytesIO(content) # 把字节流转换为文件对象
    )

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)

def delete_document_service(
    db: Session,
    document_id: int
):

    # 1. 删除Chroma向量
    delete_chunks_by_document_id(
        document_id
    )


    # 2. 删除PostgreSQL记录
    result = delete_document(
        db,
        document_id
    )


    return result


async def update_document_service(
    db: Session,
    document_id: int,
    file: UploadFile
):

    document = get_document_by_id(
        db,
        document_id
    )


    if not document:
        return None


    try:

        file_size = _get_upload_file_size(
            file
        )


        # 1. 先准备新文件数据，不修改数据库和Chroma
        prepared_data = await _prepare_document_data(
            file,
            document_id
        )


        try:

            # 2. 新数据准备成功后，替换旧的Chroma向量
            delete_chunks_by_document_id(
                document_id
            )


            save_chunks(
                prepared_data["embedded_chunks"]
            )


        except Exception:

            # 如果删除成功但保存失败，旧向量可能已经丢失。
            # 此时不修改PostgreSQL文档状态，保留原数据库记录。
            raise


        # 3. Chroma替换成功后，更新PostgreSQL文档记录
        updated_document = update_document(
            db,
            document_id,
            file.filename,
            file.content_type,
            file_size
        )


        if not updated_document:
            raise RuntimeError(
                "文档更新失败"
            )


        # 4. 更新状态为completed
        update_document_status(
            db,
            document_id,
            "completed"
        )

        logger.info(
            "Document update completed document_id=%s chunk_count=%s",
            document_id,
            len(prepared_data["chunks"]),
        )


        return {
            "filename": file.filename,
            "text_length": len(prepared_data["text"]),
            "chunk_count": len(prepared_data["chunks"]),
            "document_id": document_id
        }


    except Exception:

        # 更新预处理失败时，旧向量和旧数据库记录都不会被主动修改。
        # 更新阶段不把旧文档标记为failed，避免覆盖其原有状态。
        raise

def get_document_detail(
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

