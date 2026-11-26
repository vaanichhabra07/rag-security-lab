
import sys
import json
import time

from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


def main():

    start = time.perf_counter()

    print("Loading institutional documents...")

    documents = load_documents()

    assert len(documents) == 100, (
        f"Expected 100 documents, "
        f"found {len(documents)}"
    )

    print("Documents loaded:", len(documents))

    print("\nCreating text chunks...")

    chunks = chunk_documents(

        documents,

        chunk_size=150,

        overlap=30
    )

    print("Chunks generated:", len(chunks))

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    print(
        "Embedding dimension:",
        embedding_model.dimension
    )

    assert embedding_model.dimension == 384

    print("\nIndexing chunks in ChromaDB...")

    store = VectorStore()

    indexed_count = store.index_chunks(

        chunks,

        embedding_model
    )

    assert indexed_count == len(chunks)

    elapsed = time.perf_counter() - start

    report = {

        "dataset_version": "2.0-dev",

        "documents_loaded": len(documents),

        "document_domain_distribution": dict(

            Counter(
                doc.domain
                for doc in documents
            )
        ),

        "chunks_generated": len(chunks),

        "chunks_indexed": indexed_count,

        "chunk_size_words": 150,

        "chunk_overlap_words": 30,

        "embedding_model":
            "all-MiniLM-L6-v2",

        "embedding_dimension": 384,

        "vector_database": "ChromaDB",

        "indexing_time_seconds":
            round(elapsed, 2),

        "index_build_successful": True
    }

    output = (
        ROOT / "data" / "v2" / "metadata"
        / "rag_index_statistics.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.write_text(

        json.dumps(report, indent=2),

        encoding="utf-8"
    )

    print("\n", json.dumps(report, indent=2))

    print("\nRAG INDEX BUILD COMPLETED")


if __name__ == "__main__":
    main()
