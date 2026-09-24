import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma

from .config import settings
from .embeddings import get_embeddings


def build_index(source_dir: str) -> Chroma:
    loader = DirectoryLoader(source_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    store = Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=settings.persist_dir,
    )
    store.persist()
    return store


def load_index() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.persist_dir,
    )


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "documents"
    build_index(source)
    print(f"Indexed documents from {source} into '{settings.collection_name}'")
