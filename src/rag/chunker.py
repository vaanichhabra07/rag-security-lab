
from dataclasses import dataclass


@dataclass
class Chunk:

    chunk_id: str
    document_id: str
    domain: str
    text: str
    source: str
    chunk_index: int


def chunk_documents(
    documents,
    chunk_size=150,
    overlap=30
):

    if chunk_size <= 0:

        raise ValueError(
            "Chunk size must be positive"
        )

    if not 0 <= overlap < chunk_size:

        raise ValueError(
            "Invalid chunk overlap"
        )

    chunks = []

    step = chunk_size - overlap

    for document in documents:

        words = document.text.split()

        chunk_index = 0

        for start in range(
            0, len(words), step
        ):

            end = min(
                start + chunk_size,
                len(words)
            )

            text = " ".join(
                words[start:end]
            )

            if not text:
                continue

            chunks.append(

                Chunk(

                    chunk_id=(
                        f"{document.document_id}"
                        f"_CHUNK_{chunk_index:03d}"
                    ),

                    document_id=document.document_id,

                    domain=document.domain,

                    text=text,

                    source=document.source,

                    chunk_index=chunk_index
                )
            )

            chunk_index += 1

            if end == len(words):

                break

    return chunks
