from unittest.mock import AsyncMock, Mock

from app.exceptions.base import BadRequestException

import pytest


@pytest.fixture
def document_api(app):
    """Return the document router module after the isolated app import."""
    from app.api import document

    return document


def test_upload_txt_calls_service_and_returns_result(
    client,
    document_api,
    monkeypatch,
):
    process_document = AsyncMock(
        return_value={
            "filename": "notes.txt",
            "text_length": 11,
            "chunk_count": 1,
            "document_id": 7,
        }
    )
    monkeypatch.setattr(
        document_api,
        "process_document",
        process_document,
    )

    response = client.post(
        "/api/documents/upload",
        files={
            "file": ("notes.txt", b"hello world", "text/plain"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "message": "success",
        "data": {
            "filename": "notes.txt",
            "text_length": 11,
            "chunk_count": 1,
            "document_id": 7,
        },
    }

    process_document.assert_awaited_once()
    db, upload = process_document.await_args.args
    assert db is not None
    assert upload.filename == "notes.txt"
    assert upload.content_type == "text/plain"


def test_upload_rejects_unsupported_file_type(
    client,
    document_api,
    monkeypatch,
):
    process_document = AsyncMock(
        side_effect=BadRequestException("暂不支持该文件类型")
    )
    monkeypatch.setattr(
        document_api,
        "process_document",
        process_document,
    )

    response = client.post(
        "/api/documents/upload",
        files={
            "file": ("script.exe", b"not supported", "application/octet-stream"),
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "code": 400,
        "message": "暂不支持该文件类型",
        "data": None,
    }
    process_document.assert_awaited_once()
    assert process_document.await_args.args[1].content_type == (
        "application/octet-stream"
    )


def test_update_txt_calls_service_with_document_id_and_upload(
    client,
    document_api,
    monkeypatch,
):
    update_document_service = AsyncMock(
        return_value={
            "filename": "replacement.txt",
            "text_length": 11,
            "chunk_count": 1,
            "document_id": 12,
        }
    )
    monkeypatch.setattr(
        document_api,
        "update_document_service",
        update_document_service,
    )

    response = client.put(
        "/api/documents/12",
        files={
            "file": ("replacement.txt", b"new content", "text/plain"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "message": "success",
        "data": {
                "filename": "replacement.txt",
                "text_length": 11,
            "chunk_count": 1,
            "document_id": 12,
        },
    }

    update_document_service.assert_awaited_once()
    db, document_id, upload = update_document_service.await_args.args
    assert db is not None
    assert document_id == 12
    assert upload.filename == "replacement.txt"
    assert upload.content_type == "text/plain"


def test_update_returns_404_when_service_returns_no_document(
    client,
    document_api,
    monkeypatch,
):
    update_document_service = AsyncMock(return_value=None)
    monkeypatch.setattr(
        document_api,
        "update_document_service",
        update_document_service,
    )

    response = client.put(
        "/api/documents/999",
        files={
            "file": ("replacement.txt", b"new content", "text/plain"),
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "code": 404,
        "message": "文档不存在",
        "data": None,
    }
    assert update_document_service.await_args.args[1] == 999


def test_update_rejects_unsupported_file_type(
    client,
    document_api,
    monkeypatch,
):
    update_document_service = AsyncMock(
        side_effect=BadRequestException("暂不支持该文件类型")
    )
    monkeypatch.setattr(
        document_api,
        "update_document_service",
        update_document_service,
    )

    response = client.put(
        "/api/documents/12",
        files={
            "file": ("script.exe", b"not supported", "application/octet-stream"),
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "code": 400,
        "message": "暂不支持该文件类型",
        "data": None,
    }
    update_document_service.assert_awaited_once()
    assert update_document_service.await_args.args[2].filename == "script.exe"


def test_update_rejects_non_integer_document_id(
    client,
    document_api,
    monkeypatch,
):
    update_document_service = AsyncMock()
    monkeypatch.setattr(
        document_api,
        "update_document_service",
        update_document_service,
    )

    response = client.put(
        "/api/documents/not-an-int",
        files={
            "file": ("replacement.txt", b"new content", "text/plain"),
        },
    )

    assert response.status_code == 422
    update_document_service.assert_not_awaited()


def test_delete_existing_document_returns_success(
    client,
    document_api,
    monkeypatch,
):
    delete_document_service = Mock(return_value=True)
    monkeypatch.setattr(
        document_api,
        "delete_document_service",
        delete_document_service,
    )

    response = client.delete("/api/documents/12")

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "message": "success",
        "data": {
            "document_id": 12,
            "message": "删除成功",
        },
    }
    delete_document_service.assert_called_once()
    assert delete_document_service.call_args.args[1] == 12


def test_delete_returns_404_when_service_reports_missing_document(
    client,
    document_api,
    monkeypatch,
):
    delete_document_service = Mock(return_value=False)
    monkeypatch.setattr(
        document_api,
        "delete_document_service",
        delete_document_service,
    )

    response = client.delete("/api/documents/999")

    assert response.status_code == 404
    assert response.json() == {
        "code": 404,
        "message": "文档不存在",
        "data": None,
    }
    assert delete_document_service.call_args.args[1] == 999


def test_delete_rejects_non_integer_document_id(
    client,
    document_api,
    monkeypatch,
):
    delete_document_service = Mock()
    monkeypatch.setattr(
        document_api,
        "delete_document_service",
        delete_document_service,
    )

    response = client.delete("/api/documents/not-an-int")

    assert response.status_code == 422
    delete_document_service.assert_not_called()
