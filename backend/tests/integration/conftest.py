"""Fixtures for PostgreSQL-backed Document CRUD integration tests."""

import os
import sys
import types
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session


BACKEND_DIR = Path(__file__).resolve().parents[2]
TEST_ENV_FILE = BACKEND_DIR / ".env.test"
REQUIRED_DATABASE_NAME = "smartdoc_test"


def _install_import_time_doubles() -> None:
    """Prevent service imports from loading models or initializing Chroma."""
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


def _read_test_database_url() -> str:
    """Read only the dedicated test URL; never fall back to production config."""
    environment_url = os.environ.get("TEST_DATABASE_URL")
    file_values = (
        dotenv_values(TEST_ENV_FILE)
        if TEST_ENV_FILE.exists()
        else {}
    )
    configured_url = environment_url or file_values.get("TEST_DATABASE_URL")

    if not configured_url:
        pytest.fail(
            "TEST_DATABASE_URL is not configured. Create backend/.env.test "
            "from .env.test.example before running integration tests."
        )

    try:
        parsed_url = make_url(configured_url)
    except Exception:
        pytest.fail(
            "TEST_DATABASE_URL is invalid; its value was not displayed."
        )

    if parsed_url.database != REQUIRED_DATABASE_NAME:
        pytest.fail(
            "Refusing integration tests: TEST_DATABASE_URL must target "
            f"{REQUIRED_DATABASE_NAME}, never the development database."
        )

    print(f"database={parsed_url.database}")
    return configured_url


def _local_alembic_heads() -> set[str]:
    alembic_config = Config(str(BACKEND_DIR / "alembic.ini"))
    script_directory = ScriptDirectory.from_config(alembic_config)
    return set(script_directory.get_heads())


def _verify_database_ready(engine) -> None:
    try:
        with engine.connect() as connection:
            actual_database = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()
            documents_table = connection.execute(
                text("SELECT to_regclass('public.documents')")
            ).scalar_one()
            current_versions = set(
                connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalars().all()
            )
    except Exception as exc:
        pytest.fail(
            "Could not verify smartdoc_test. Create the database and run "
            f"Alembic upgrade head first ({type(exc).__name__})."
        )

    if actual_database != REQUIRED_DATABASE_NAME:
        pytest.fail(
            "Refusing integration tests: connected database is not "
            f"{REQUIRED_DATABASE_NAME}."
        )

    if documents_table != "documents":
        pytest.fail(
            "smartdoc_test is missing public.documents. Run "
            "python -m alembic upgrade head against the test URL."
        )

    if current_versions != _local_alembic_heads():
        pytest.fail(
            "smartdoc_test Alembic version is not at head. Run "
            "python -m alembic upgrade head against the test URL."
        )


@pytest.fixture(scope="session")
def test_database_url() -> str:
    return _read_test_database_url()


@pytest.fixture(scope="session")
def test_engine(test_database_url: str):
    parsed_url: URL = make_url(test_database_url)
    engine = create_engine(parsed_url, pool_pre_ping=True)
    _verify_database_ready(engine)

    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(test_engine):
    """Run each CRUD test in a transaction rolled back at test completion."""
    with test_engine.connect() as connection:
        transaction = connection.begin()
        session = Session(
            bind=connection,
            join_transaction_mode="create_savepoint",
        )

        try:
            yield session
        finally:
            session.close()
            transaction.rollback()
