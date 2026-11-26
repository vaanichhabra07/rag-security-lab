
import time

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        embedding_model=None,
        vector_store=None
    ):

        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else EmbeddingModel()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )

    def retrieve(
        self,
        query,
        top_k=3,
        domain=None
    ):

        start = time.perf_counter()

        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        results = self.vector_store.search(

            query_embedding=query_embedding,

            top_k=top_k,

            domain=domain
        )

        retrieved_chunks = []

        for rank, (
            chunk_id,
            text,
            metadata,
            distance
        ) in enumerate(

            zip(

                results["ids"][0],

                results["documents"][0],

                results["metadatas"][0],

                results["distances"][0]
            ),

            start=1
        ):

            retrieved_chunks.append({

                "rank": rank,

                "chunk_id": chunk_id,

                "document_id":
                    metadata["document_id"],

                "domain":
                    metadata["domain"],

                "source":
                    metadata["source"],

                "text": text,

                "distance": float(distance),

                "similarity": float(
                    1 - distance
                )
            })

        latency = (
            time.perf_counter() - start
        ) * 1000

        return {

            "query": query,

            "top_k": top_k,

            "retrieved_chunks":
                retrieved_chunks,

            "retrieval_latency_ms":
                round(latency, 2)
        }
