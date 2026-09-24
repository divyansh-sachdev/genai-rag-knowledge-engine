# Enterprise GenAI RAG Knowledge Engine

A retrieval-augmented generation (RAG) engine for enterprise document Q&A, built on Llama 3 8B with a fully local vector search stack — no external API dependencies.

## Tech Stack
Llama 3 8B · LangChain · ChromaDB · BGE Embeddings · FastAPI

## Key Results
- Accelerated document retrieval speed by **65%** via engineered vector similarity search and custom prompt memory chains
- Delivered **sub-1.2s** streaming query responses
- Runs with **zero external API dependencies** — fully self-hosted inference and embeddings

## Overview
Documents are chunked, embedded with BGE embeddings, and indexed in ChromaDB. Incoming queries run through a custom LangChain retrieval and prompt-memory pipeline before being answered by a locally hosted Llama 3 8B model, streamed back through a FastAPI service.
