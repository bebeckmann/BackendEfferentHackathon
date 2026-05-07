"""FastAPI service — exposes the RAG retrieval pipeline as an HTTP endpoint."""
from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel
from qdrant_client import QdrantClient

from retrieve import (
    COLLECTION, EMBED_DIM, OPENROUTER_BASE, QDRANT_PATH,
    ask, build_chain, load_doc_store,
)

load_dotenv()

# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    grounding_images: list[str]  # base64-encoded PNGs, one per cited page

# ── App lifecycle ─────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ["OPENROUTER_API_KEY"]
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        dimensions=EMBED_DIM,
    )
    qdrant = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION,
        embedding=embeddings,
    )
    _state["chain"] = build_chain(vector_store)
    _state["doc_store"] = load_doc_store()
    _state["qdrant"] = qdrant
    yield
    qdrant.close()


app = FastAPI(title="Literature RAG API", version="0.1.0", lifespan=lifespan)

# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=QueryResponse)
async def ask_endpoint(req: QueryRequest):
    """Answer a question using the indexed literature and return cited grounding images."""
    try:
        answer, images = ask(req.question, _state["chain"], _state["doc_store"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    images_b64: list[str] = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode())

    return QueryResponse(answer=answer, grounding_images=images_b64)


@app.get("/health")
async def health():
    return {"status": "ok"}
