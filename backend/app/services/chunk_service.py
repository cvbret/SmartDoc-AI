def split_text(
    text: str,
    filename: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:

    if not text:
        return []

    chunks = []

    if len(text) <= chunk_size:
        chunks.append(
            {
                "content": text,
                "metadata": {
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
                    "filename": filename,
                    "chunk_id": chunk_id
                }
            }
        )


        chunk_id += 1

        start = end - overlap


    return chunks