from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from .config import settings


def get_embeddings() -> HuggingFaceBgeEmbeddings:
    """BGE embeddings, run locally — no external embedding API calls."""
    return HuggingFaceBgeEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
