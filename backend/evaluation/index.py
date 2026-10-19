"""A disposable index whose record IDs are the dataset's stable context IDs."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from uuid import uuid4


@contextmanager
def evaluation_index(
    contexts: dict[str, str], embed: Callable[[str], list[float]],
) -> Iterator:
    import chromadb

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=f"rag-evaluation-{uuid4().hex}", embedding_function=None,
    )
    try:
        if contexts:
            collection.add(
                ids=list(contexts), documents=list(contexts.values()),
                embeddings=[embed(text) for text in contexts.values()],
                metadatas=[{"evaluation_context_id": context_id} for context_id in contexts],
            )
        yield collection
    finally:
        client.delete_collection(collection.name)


def query_index(collection, embedding: list[float], retrieval_k: int) -> dict:
    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be positive")
    effective_k = min(retrieval_k, collection.count())
    if effective_k == 0:
        return {}
    return collection.query(
        query_embeddings=[embedding], n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )
