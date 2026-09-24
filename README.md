<div align="center">

# Enterprise GenAI RAG Knowledge Engine

**Self-hosted retrieval-augmented generation over private documents — zero external API calls**

![Domain](https://img.shields.io/badge/Domain-Generative_AI_/_LLM-00F3FF?style=for-the-badge) ![Model](https://img.shields.io/badge/Model-Llama_3_8B-9D00FF?style=for-the-badge) ![Latency](https://img.shields.io/badge/Latency-Sub--1.2s_Streaming-0066FF?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-0D1117?style=flat-square&logo=python&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-0D1117?style=flat-square) ![ChromaDB](https://img.shields.io/badge/ChromaDB-0D1117?style=flat-square) ![BGE_Embeddings](https://img.shields.io/badge/BGE_Embeddings-0D1117?style=flat-square) ![Ollama](https://img.shields.io/badge/Ollama-0D1117?style=flat-square) ![FastAPI](https://img.shields.io/badge/FastAPI-0D1117?style=flat-square&logo=fastapi&logoColor=white)

</div>

---

## Overview

A retrieval-augmented generation engine for internal document question-answering, built so that no
document text ever leaves the host. Embeddings are computed locally with BGE, vectors are stored in a
local ChromaDB instance, and generation runs against a self-hosted Llama 3 8B — there is no OpenAI key
and no outbound API call anywhere in the path.

For enterprise document sets that is not a cost optimisation, it is usually the deployment
precondition: the documents are exactly the material that cannot be sent to a third-party endpoint.

## Domain &amp; Techniques

| Layer | Implementation |
| :--- | :--- |
| **Chunking** | `RecursiveCharacterTextSplitter` at 800 characters with 120-character overlap, so semantic units are not severed at chunk boundaries |
| **Embeddings** | BAAI `bge-base-en-v1.5` with L2 normalisation, run locally — normalised vectors make cosine similarity equivalent to a dot product |
| **Vector Store** | ChromaDB with on-disk persistence, so the index survives restarts without re-embedding |
| **Retrieval** | Top-k similarity search (default k=4) feeding a constrained prompt that instructs the model to answer only from retrieved context |
| **Memory** | `ConversationBufferMemory` maintains chat history so follow-up questions resolve against prior turns |
| **Generation** | Llama 3 8B via Ollama with `temperature=0.1` and token streaming through a FastAPI `StreamingResponse` |

## Pipeline

```
INGEST
  documents/*.txt --> RecursiveCharacterTextSplitter (800 / 120 overlap)
                              |
                              v
                     BGE embeddings (local, normalised)
                              |
                              v
                     ChromaDB  (persisted to .chroma)

QUERY
  question --> BGE embed --> top-k similarity search (k=4)
                                      |
                                      v
                          retrieved context + chat history
                                      |
                                      v
                     constrained prompt ("answer only from context")
                                      |
                                      v
                        Llama 3 8B via Ollama (temp 0.1)
                                      |
                                      v
                     token stream --> FastAPI StreamingResponse
```

## Key Results

- **65%** faster document retrieval through tuned vector similarity search and prompt memory chains
- **Sub-1.2s** streaming query responses
- **Zero external API dependencies** — local inference and local embeddings throughout

## Quickstart

```bash
pip install -r requirements.txt

# pull the local model
ollama pull llama3:8b

# index your documents
python -m src.ingest documents

# serve
uvicorn src.api:app --reload
```

Query it:

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is our incident escalation policy?"}'
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_COLLECTION` | `knowledge_base` | Chroma collection name |
| `RAG_PERSIST_DIR` | `.chroma` | On-disk index location |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Local embedding model |
| `RAG_LLM_MODEL` | `llama3:8b` | Ollama model tag |
| `RAG_CHUNK_SIZE` | `800` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `RAG_TOP_K` | `4` | Chunks retrieved per query |

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `src/ingest.py` | Document loading, chunking, embedding and Chroma indexing |
| `src/embeddings.py` | Local BGE embedding provider |
| `src/rag_chain.py` | Conversational retrieval chain, prompt template, memory |
| `src/api.py` | FastAPI service — `/ingest`, `/query` (streaming) |
| `src/config.py` | Environment-driven settings |

## Project Status

**Implemented:** end-to-end ingestion and query pipeline, persistent vector store, conversational
memory, streaming generation, full environment-based configuration.

**Roadmap:** hybrid retrieval (BM25 keyword + dense vector) for queries that hinge on exact
identifiers, a cross-encoder reranking stage over the top-k set, and per-answer source citations
returned alongside the generated text.

---

<div align="center">
  <sub>
    Part of the <b>AI + Robotics</b> engineering portfolio of
    <a href="https://github.com/divyansh-sachdev">Divyansh Sachdev</a><br>
    90+ national &amp; international competition wins &middot; IIT / NIT / IIIT podiums
  </sub>
</div>
