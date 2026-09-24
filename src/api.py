from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .ingest import build_index
from .rag_chain import build_chain

app = FastAPI(title="Enterprise GenAI RAG Knowledge Engine")
_chain = None


class Query(BaseModel):
    question: str


class IngestRequest(BaseModel):
    source_dir: str = "documents"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    build_index(req.source_dir)
    return {"status": "indexed", "source": req.source_dir}


def _get_chain():
    global _chain
    if _chain is None:
        _chain = build_chain()
    return _chain


@app.post("/query")
def query(q: Query):
    chain = _get_chain()

    def token_stream():
        for chunk in chain.stream({"question": q.question}):
            token = chunk.get("answer", "")
            if token:
                yield token

    return StreamingResponse(token_stream(), media_type="text/plain")
