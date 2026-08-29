import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.models.document import Document
from app.services import document_service
from app.services.document_crud import (
    create_document,
    get_document_by_id,
)


pytestmark = pytest.mark.integration


def _upload_file(
    filename: str = "document.txt",
    content: bytes = b"service test content",
    content_type: str = "text/plain",
) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


def _patch_process_dependencies(monkeypatch):
    parsed_text = "parsed service text"
    state = {
        "chunks": None,
        "embedded_chunks": None,
    }

    parse_document = AsyncMock(return_value=parsed_text)

    def split_text_impl(text, filename, document_id):
        state["chunks"] = [
            {
                "content": text,
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": 0,
                },
            }
        ]
        return state["chunks"]

    split_text = Mock(side_effect=split_text_impl)

    def embed_chunks_impl(chunks):
        state["embedded_chunks"] = [
            {
                "content": chunk["content"],
                "embedding": [0.1, 0.2, 0.3],
                "metadata": chunk["metadata"],
            }
            for chunk in chunks
        ]
        return state["embedded_chunks"]

    embed_chunks = Mock(side_effect=embed_chunks_impl)
    save_chunks = Mock()

    monkeypatch.setattr(document_service, "parse_document", parse_document)
    monkeypatch.setattr(document_service, "split_text", split_text)
    monkeypatch.setattr(document_service, "embed_chunks", embed_chunks)
    monkeypatch.setattr(document_service, "save_chunks", save_chunks)

    return (
        parse_document,
        split_text,
        embed_chunks,
        save_chunks,
        state,
        parsed_text,
    )


def test_process_document_success_persists_completed_document(
    db_session,
    monkeypatch,
):
    (
        parse_document,
        split_text,
        embed_chunks,
        save_chunks,
        state,
        parsed_text,
    ) = _patch_process_dependencies(monkeypatch)
    upload = _upload_file(content=b"hello service")

    result = asyncio.run(
        document_service.process_document(db_session, upload)
    )
    persisted = get_document_by_id(db_session, result["document_id"])

    assert persisted is not None
    assert result == {
        "filename": "document.txt",
        "text_length": len(parsed_text),
        "chunk_count": 1,
        "document_id": persisted.id,
    }
    assert persisted.filename == "document.txt"
    assert persisted.file_type == "text/plain"
    assert persisted.file_size == len(b"hello service")
    assert persisted.status == "completed"
    parse_document.assert_awaited_once_with(upload)
    split_text.assert_called_once_with(
        parsed_text,
        "document.txt",
        persisted.id,
    )
    embed_chunks.assert_called_once_with(state["chunks"])
    save_chunks.assert_called_once_with(state["embedded_chunks"])
    assert state["embedded_chunks"][0]["metadata"]["document_id"] == (
        persisted.id
    )


def test_process_document_failure_marks_document_failed(
    db_session,
    monkeypatch,
):
    parsed_text = "parsed failure text"
    parse_document = AsyncMock(return_value=parsed_text)
    split_text = Mock(
        return_value=[
            {
                "content": parsed_text,
                "metadata": {
                    "document_id": 0,
                    "filename": "failed-document.txt",
                    "chunk_id": 0,
                },
            }
        ]
    )
    embed_chunks = Mock(side_effect=RuntimeError("embedding failed"))
    save_chunks = Mock()
    monkeypatch.setattr(document_service, "parse_document", parse_document)
    monkeypatch.setattr(document_service, "split_text", split_text)
    monkeypatch.setattr(document_service, "embed_chunks", embed_chunks)
    monkeypatch.setattr(document_service, "save_chunks", save_chunks)
    upload = _upload_file(
        filename="failed-document.txt",
        content=b"failure content",
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        asyncio.run(
            document_service.process_document(db_session, upload)
        )

    persisted = (
        db_session.query(Document)
        .filter(Document.filename == "failed-document.txt")
        .order_by(Document.id.desc())
        .first()
    )

    assert persisted is not None
    assert persisted.status == "failed"
    parse_document.assert_awaited_once_with(upload)
    embed_chunks.assert_called_once_with(split_text.return_value)
    save_chunks.assert_not_called()


def test_update_document_service_success_updates_database_and_vectors(
    db_session,
    monkeypatch,
):
    created = create_document(
        db_session,
        "before.txt",
        "text/plain",
        10,
    )
    (
        parse_document,
        split_text,
        embed_chunks,
        save_chunks,
        state,
        parsed_text,
    ) = _patch_process_dependencies(monkeypatch)
    delete_chunks = Mock()
    monkeypatch.setattr(
        document_service,
        "delete_chunks_by_document_id",
        delete_chunks,
    )
    upload = _upload_file(
        filename="after.pdf",
        content=b"replacement content",
        content_type="application/pdf",
    )

    result = asyncio.run(
        document_service.update_document_service(
            db_session,
            created.id,
            upload,
        )
    )
    persisted = get_document_by_id(db_session, created.id)

    assert result == {
        "filename": "after.pdf",
        "text_length": len(parsed_text),
        "chunk_count": 1,
        "document_id": created.id,
    }
    assert persisted is not None
    assert persisted.id == created.id
    assert persisted.filename == "after.pdf"
    assert persisted.file_type == "application/pdf"
    assert persisted.file_size == len(b"replacement content")
    assert persisted.status == "completed"
    delete_chunks.assert_called_once_with(created.id)
    save_chunks.assert_called_once_with(state["embedded_chunks"])
    assert state["embedded_chunks"][0]["metadata"]["document_id"] == (
        created.id
    )
    parse_document.assert_awaited_once_with(upload)
    split_text.assert_called_once_with(
        parsed_text,
        "after.pdf",
        created.id,
    )
    embed_chunks.assert_called_once_with(state["chunks"])


def test_update_document_service_returns_none_for_missing_document(
    db_session,
    monkeypatch,
):
    parse_document = AsyncMock()
    monkeypatch.setattr(document_service, "parse_document", parse_document)
    upload = _upload_file()

    result = asyncio.run(
        document_service.update_document_service(
            db_session,
            -1,
            upload,
        )
    )

    assert result is None
    parse_document.assert_not_awaited()


def test_delete_document_service_deletes_database_row_and_vectors(
    db_session,
    monkeypatch,
):
    created = create_document(
        db_session,
        "to-delete.txt",
        "text/plain",
        12,
    )
    events = []

    def get_document(*args):
        events.append("get")
        return get_document_by_id(*args)

    def delete_chunks(document_id):
        events.append("chroma")

    def delete_from_database(*args):
        events.append("db")
        from app.services.document_crud import delete_document

        return delete_document(*args)

    monkeypatch.setattr(
        document_service,
        "delete_chunks_by_document_id",
        delete_chunks,
    )
    monkeypatch.setattr(
        document_service,
        "get_document_by_id",
        get_document,
    )
    monkeypatch.setattr(
        document_service,
        "delete_document",
        delete_from_database,
    )

    result = document_service.delete_document_service(
        db_session,
        created.id,
    )

    assert result is True
    assert events == ["get", "chroma", "db"]
    assert get_document_by_id(db_session, created.id) is None


def test_delete_document_service_returns_false_for_missing_document(
    db_session,
    monkeypatch,
):
    delete_chunks = Mock()
    delete_from_database = Mock()
    monkeypatch.setattr(
        document_service,
        "delete_chunks_by_document_id",
        delete_chunks,
    )
    monkeypatch.setattr(
        document_service,
        "delete_document",
        delete_from_database,
    )

    result = document_service.delete_document_service(db_session, -1)

    assert result is False
    delete_chunks.assert_not_called()
    delete_from_database.assert_not_called()


def test_get_document_detail_returns_document_and_none_for_missing(db_session):
    created = create_document(
        db_session,
        "detail.txt",
        "text/plain",
        20,
    )

    found = document_service.get_document_detail(db_session, created.id)

    assert found is not None
    assert found.id == created.id
    assert found.filename == "detail.txt"
    assert document_service.get_document_detail(db_session, -1) is None
