import pytest

from app.services.document_crud import (
    create_document,
    delete_document,
    get_document_by_id,
    get_documents,
    update_document,
    update_document_status,
)


pytestmark = pytest.mark.integration


def test_create_document(db_session):
    document = create_document(
        db_session,
        "handbook.txt",
        "text/plain",
        2048,
    )

    assert document.id is not None
    assert document.filename == "handbook.txt"
    assert document.file_type == "text/plain"
    assert document.file_size == 2048
    assert document.status == "processing"


def test_get_document_by_id_returns_document_and_none_for_missing(db_session):
    created = create_document(
        db_session,
        "lookup.pdf",
        "application/pdf",
        4096,
    )

    found = get_document_by_id(db_session, created.id)

    assert found is not None
    assert found.id == created.id
    assert found.filename == "lookup.pdf"
    assert get_document_by_id(db_session, -1) is None


def test_get_documents_paginates_in_descending_id_order(db_session):
    created_documents = [
        create_document(
            db_session,
            f"document-{index}.txt",
            "text/plain",
            index,
        )
        for index in range(1, 6)
    ]

    first_page, total = get_documents(
        db_session,
        page=1,
        page_size=2,
    )
    second_page, _ = get_documents(
        db_session,
        page=2,
        page_size=2,
    )
    third_page, _ = get_documents(
        db_session,
        page=3,
        page_size=2,
    )

    expected_descending_ids = [
        document.id for document in reversed(created_documents)
    ]
    assert total == 5
    assert [document.id for document in first_page] == (
        expected_descending_ids[:2]
    )
    assert [document.id for document in second_page] == (
        expected_descending_ids[2:4]
    )
    assert [document.id for document in third_page] == (
        expected_descending_ids[4:]
    )


def test_get_documents_filters_by_status(db_session):
    completed = create_document(
        db_session,
        "completed.txt",
        "text/plain",
        1,
    )
    processing = create_document(
        db_session,
        "processing.txt",
        "text/plain",
        2,
    )
    failed = create_document(
        db_session,
        "failed.txt",
        "text/plain",
        3,
    )
    update_document_status(db_session, completed.id, "completed")
    update_document_status(db_session, processing.id, "processing")
    update_document_status(db_session, failed.id, "failed")

    documents, total = get_documents(
        db_session,
        status="completed",
    )

    assert total == 1
    assert [document.id for document in documents] == [completed.id]
    assert documents[0].status == "completed"


def test_update_document_changes_fields_and_status_helper(db_session):
    created = create_document(
        db_session,
        "before.txt",
        "text/plain",
        10,
    )

    updated = update_document(
        db_session,
        created.id,
        "after.pdf",
        "application/pdf",
        20,
    )

    assert updated is not None
    assert updated.filename == "after.pdf"
    assert updated.file_type == "application/pdf"
    assert updated.file_size == 20
    assert updated.status == "processing"

    update_document_status(db_session, created.id, "completed")
    refreshed = get_document_by_id(db_session, created.id)

    assert refreshed is not None
    assert refreshed.filename == "after.pdf"
    assert refreshed.file_type == "application/pdf"
    assert refreshed.file_size == 20
    assert refreshed.status == "completed"


def test_delete_document_removes_document_and_returns_false_when_missing(
    db_session,
):
    created = create_document(
        db_session,
        "to-delete.txt",
        "text/plain",
        12,
    )

    assert delete_document(db_session, created.id) is True
    assert get_document_by_id(db_session, created.id) is None
    assert delete_document(db_session, created.id) is False
