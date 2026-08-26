from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def document_api(app):
    """Return the document router module after the isolated app import."""
    from app.api import document

    return document


@pytest.fixture
def sample_document():
    """Provide an ORM-like object accepted by DocumentResponse."""
    return SimpleNamespace(
        id=7,
        filename="employee-guide.pdf",
        file_type="application/pdf",
        file_size=2048,
        status="completed",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )


def test_list_documents_returns_expected_payload(
    client,
    document_api,
    monkeypatch,
    sample_document,
):
    get_documents = Mock(return_value=([sample_document], 1))
    monkeypatch.setattr(document_api, "get_documents", get_documents)

    response = client.get("/api/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "success"
    assert body["data"]["items"] == [
        {
            "id": 7,
            "filename": "employee-guide.pdf",
            "file_type": "application/pdf",
            "file_size": 2048,
            "status": "completed",
            "created_at": "2026-01-02T03:04:05Z",
        }
    ]
    assert body["data"]["page"] == 1
    assert body["data"]["page_size"] == 10
    assert body["data"]["total"] == 1
    assert body["data"]["total_pages"] == 1

    assert get_documents.call_args.args[1:] == (1, 10, None)


def test_list_documents_passes_pagination_and_status(
    client,
    document_api,
    monkeypatch,
):
    get_documents = Mock(return_value=([], 12))
    monkeypatch.setattr(document_api, "get_documents", get_documents)

    response = client.get(
        "/api/documents?page=2&page_size=5&status=completed"
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "items": [],
        "page": 2,
        "page_size": 5,
        "total": 12,
        "total_pages": 3,
    }
    assert get_documents.call_args.args[1:] == (2, 5, "completed")


@pytest.mark.parametrize(
    "query",
    [
        "?page=0",
        "?page_size=0",
        "?page_size=101",
    ],
)
def test_list_documents_rejects_invalid_pagination(
    client,
    document_api,
    monkeypatch,
    query,
):
    get_documents = Mock()
    monkeypatch.setattr(document_api, "get_documents", get_documents)

    response = client.get(f"/api/documents{query}")

    assert response.status_code == 422
    get_documents.assert_not_called()


def test_get_document_detail_returns_document(
    client,
    document_api,
    monkeypatch,
    sample_document,
):
    get_document_detail = Mock(return_value=sample_document)
    monkeypatch.setattr(
        document_api,
        "get_document_detail",
        get_document_detail,
    )

    response = client.get("/api/documents/7")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "success"
    assert body["data"]["id"] == 7
    assert body["data"]["filename"] == "employee-guide.pdf"
    assert body["data"]["file_type"] == "application/pdf"
    assert body["data"]["status"] == "completed"
    assert get_document_detail.call_args.args[1] == 7


def test_get_document_detail_returns_404_when_missing(
    client,
    document_api,
    monkeypatch,
):
    get_document_detail = Mock(return_value=None)
    monkeypatch.setattr(
        document_api,
        "get_document_detail",
        get_document_detail,
    )

    response = client.get("/api/documents/999")

    assert response.status_code == 404
    assert response.json() == {
        "code": 404,
        "message": "文档不存在",
        "data": None,
    }
    assert get_document_detail.call_args.args[1] == 999


def test_get_document_detail_rejects_non_integer_id(
    client,
    document_api,
    monkeypatch,
):
    get_document_detail = Mock()
    monkeypatch.setattr(
        document_api,
        "get_document_detail",
        get_document_detail,
    )

    response = client.get("/api/documents/not-an-int")

    assert response.status_code == 422
    get_document_detail.assert_not_called()
