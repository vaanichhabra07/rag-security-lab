
from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        self.dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

    def embed_documents(self, texts):

        if not texts:

            return []

        embeddings = self.model.encode(

            texts,

            batch_size=32,

            normalize_embeddings=True,

            show_progress_bar=False
        )

        return embeddings.tolist()

    def embed_query(self, query):

        if not query.strip():

            raise ValueError(
                "Query cannot be empty"
            )

        embedding = self.model.encode(

            query,

            normalize_embeddings=True,

            show_progress_bar=False
        )

        return embedding.tolist()
