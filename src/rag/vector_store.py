
from pathlib import Path

import chromadb


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    ROOT / "data" / "v2" / "vector_store"
)

COLLECTION_NAME = "institutional_corpus_v2"


class VectorStore:

    def __init__(self):

        DB_PATH.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = (
            self.client.get_or_create_collection(

                name=COLLECTION_NAME,

                metadata={
                    "hnsw:space": "cosine"
                },

                embedding_function=None
            )
        )

    def index_chunks(
        self,
        chunks,
        embedding_model,
        batch_size=64
    ):

        if not chunks:

            raise ValueError(
                "No chunks available for indexing"
            )

        for start in range(
            0, len(chunks), batch_size
        ):

            batch = chunks[
                start:start + batch_size
            ]

            texts = [
                chunk.text
                for chunk in batch
            ]

            embeddings = (
                embedding_model.embed_documents(
                    texts
                )
            )

            self.collection.upsert(

                ids=[
                    chunk.chunk_id
                    for chunk in batch
                ],

                documents=texts,

                embeddings=embeddings,

                metadatas=[

                    {
                        "document_id":
                            chunk.document_id,

                        "domain":
                            chunk.domain,

                        "source":
                            chunk.source,

                        "chunk_index":
                            chunk.chunk_index
                    }

                    for chunk in batch
                ]
            )

        # Remove stale chunks after successful indexing.

        current_ids = {
            chunk.chunk_id
            for chunk in chunks
        }

        stored_ids = set(
            self.collection.get(
                include=[]
            )["ids"]
        )

        stale_ids = list(
            stored_ids - current_ids
        )

        if stale_ids:

            self.collection.delete(
                ids=stale_ids
            )

        return self.collection.count()

    def count(self):

        return self.collection.count()

    def search(
        self,
        query_embedding,
        top_k=3,
        domain=None
    ):

        if top_k <= 0:

            raise ValueError(
                "top_k must be positive"
            )

        count = self.collection.count()

        if count == 0:

            raise RuntimeError(
                "Vector database is empty. "
                "Build the index first."
            )

        arguments = {

            "query_embeddings": [
                query_embedding
            ],

            "n_results": min(top_k, count),

            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        if domain:

            arguments["where"] = {
                "domain": domain
            }

        results = self.collection.query(
            **arguments
        )

        return results
