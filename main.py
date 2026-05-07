"""FastAPI service — RAG pipeline exposed as an HTTP endpoint."""
from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retrieve import _get_rag, ask as rag_ask

load_dotenv()

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,https://www.sepsis-analysis.online,https://sepsis-analysis.online",
).split(",")
FRONTEND_ORIGINS = [o.strip() for o in FRONTEND_ORIGINS if o.strip()]

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EVIDENCE_DIR = STATIC_DIR / "evidence"

# ── Schemas ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str


class EvidenceImage(BaseModel):
    url: str
    caption: str
    source: str
    kind: str  # "grounding" | "uploaded"


class AgentResponse(BaseModel):
    answer: str
    images: list[EvidenceImage]
    warnings: list[str]


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(_get_rag)  # warm up Qdrant + embeddings at startup
    yield


app = FastAPI(title="Sepsis Research Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/api/chat", response_model=AgentResponse)
async def chat(
    message: Annotated[str, Form()],
    session_id: Annotated[str | None, Form()] = None,
    images: Annotated[list[UploadFile] | None, File()] = None,
):
    """
    Accepts a text message (and optional uploaded images / session_id).
    Returns the RAG answer plus grounding page images as data URLs.
    """
    _ = session_id, images  # accepted for API compatibility, unused by RAG pipeline

    def _run(question: str):
        chain, doc_store = _get_rag()
        return rag_ask(question, chain, doc_store)

    try:
        answer, grounding_images, cited_chunks = await run_in_threadpool(_run, message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    evidence: list[EvidenceImage] = []
    for img, chunk in zip(grounding_images, cited_chunks):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        evidence.append(
            EvidenceImage(
                url=f"data:image/png;base64,{b64}",
                caption=f"[{chunk.num}] {chunk.pdf_name} — page {chunk.page_no}",
                source=chunk.pdf_name,
                kind="grounding",
            )
        )

    return AgentResponse(
        answer=answer,
        images=evidence,
        warnings=[
            "Research-use only. Not for clinical diagnosis, triage, or treatment decisions."
        ],
    )
