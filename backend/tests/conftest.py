"""Shared fixtures for isolated FastAPI API tests."""

import sys
import types

import pytest
from fastapi.testclient import TestClient


def _install_import_time_doubles() -> None:
    """Avoid model and Chroma initialization while importing the real app.

    ``app.main`` imports all API routers at module import time. The health
    endpoint does not use SentenceTransformer or Chroma, so test-only module
    doubles keep this smoke test local and lightweight without changing
    production initialization behavior.
    """
    if "sentence_transformers" not in sys.modules:
        sentence_transformers = types.ModuleType("sentence_transformers")

        class FakeSentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

        sentence_transformers.SentenceTransformer = FakeSentenceTransformer
        sys.modules["sentence_transformers"] = sentence_transformers

    if "chromadb" not in sys.modules:
        chromadb = types.ModuleType("chromadb")

        class FakeCollection:
            pass

        class FakePersistentClient:
            def __init__(self, *args, **kwargs):
                self.collection = FakeCollection()

            def get_or_create_collection(self, *args, **kwargs):
                return self.collection

        chromadb.PersistentClient = FakePersistentClient
        sys.modules["chromadb"] = chromadb


_install_import_time_doubles()


@pytest.fixture(scope="session")
def app():
    """Return the project's real FastAPI application instance."""
    _install_import_time_doubles()

    from app.main import app as fastapi_app

    return fastapi_app


@pytest.fixture
def client(app):
    """Provide a TestClient without starting Uvicorn."""
    from app.db.session import get_db

    missing = object()
    previous_override = app.dependency_overrides.get(get_db, missing)

    def override_get_db():
        yield object()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        if previous_override is missing:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = previous_override
