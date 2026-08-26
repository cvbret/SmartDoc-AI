import logging


logger = logging.getLogger(__name__)


def split_text(
    text: str,
    filename: str,
    document_id: int,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:

    logger.debug(
        "Splitting document_id=%s filename=%s chunk_size=%s overlap=%s",
        document_id,
        filename,
        chunk_size,
        overlap,
    )

    if not text:
        return []

    chunks = []

    if len(text) <= chunk_size:
        chunks.append(
            {
                "content": text,
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": 0
                }
            }
        )

        return chunks


    start = 0
    chunk_id = 0


    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]


        chunks.append(
            {
                "content": chunk,
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": chunk_id
                }
            }
        )


        chunk_id += 1

        start = end - overlap


    return chunks
