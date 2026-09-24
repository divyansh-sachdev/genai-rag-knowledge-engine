import os
from dataclasses import dataclass


@dataclass
class Settings:
    collection_name: str = os.getenv("RAG_COLLECTION", "knowledge_base")
    persist_dir: str = os.getenv("RAG_PERSIST_DIR", ".chroma")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "llama3:8b")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))


settings = Settings()
