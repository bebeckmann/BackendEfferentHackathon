"""FastAPI service — RAG pipeline exposed as an HTTP endpoint."""
from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from index import index as run_index, read_index_entries
from retrieve import (
    _get_rag,
    ask,
    delete_all_documents,
    delete_document,
    find_doc_hash_by_title,
    invalidate_rag_cache,
    list_indexed_documents,
)

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
    data: str   # base64-encoded PNG as a data URI
    caption: str
    source: str
    kind: str  # "grounding" | "uploaded"


class AgentResponse(BaseModel):
    answer: str
    images: list[EvidenceImage]
    warnings: list[str]


class IndexResponse(BaseModel):
    indexed: list[str]
    total_chunks: int


class DocumentEntry(BaseModel):
    doc_hash: str
    paper_title: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentEntry]


class IndexEntryResponse(BaseModel):
    number: int
    total: int
    entry: str


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
        answer, grounding_images, _ = ask(question, chain, doc_store)
        return answer, grounding_images

    try:
        answer, grounding_images = await run_in_threadpool(_run, message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    evidence: list[EvidenceImage] = []
    for i, img in enumerate(grounding_images, start=1):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        evidence.append(
            EvidenceImage(
                data=f"data:image/png;base64,{b64}",
                caption=f"Source {i}",
                source="literature",
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


@app.delete("/api/documents/all", status_code=200)
async def delete_all_documents_endpoint():
    """Delete every indexed document, all vector chunks, and all doc-store JSON files."""
    count = await run_in_threadpool(delete_all_documents)
    invalidate_rag_cache()
    return {"deleted": count}


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed PDFs with their doc hash (UUID) and paper title."""
    docs = await run_in_threadpool(list_indexed_documents)
    return DocumentListResponse(documents=[DocumentEntry(**d) for d in docs])


@app.delete("/api/documents", status_code=200)
async def delete_document_endpoint(
    doc_hash: Annotated[str | None, Query()] = None,
    name: Annotated[str | None, Query()] = None,
):
    """Delete a document and all its vector chunks by doc_hash or paper title."""
    if not doc_hash and not name:
        raise HTTPException(status_code=422, detail="Provide either 'doc_hash' or 'name'.")
    if doc_hash and name:
        raise HTTPException(status_code=422, detail="Provide only one of 'doc_hash' or 'name', not both.")

    resolved_hash = doc_hash
    if name and not resolved_hash:
        resolved_hash = await run_in_threadpool(find_doc_hash_by_title, name)
        if not resolved_hash:
            raise HTTPException(status_code=404, detail=f"No document found with name {name!r}.")

    deleted = await run_in_threadpool(delete_document, resolved_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No document found with doc_hash {resolved_hash!r}.")

    invalidate_rag_cache()
    return {"deleted": resolved_hash}


@app.get("/api/index/entries")
async def get_full_index():
    """Return the entire index.md file as a markdown string."""
    entries = await run_in_threadpool(read_index_entries)
    if not entries:
        raise HTTPException(status_code=404, detail="index.md is empty or does not exist.")
    return {"total": len(entries), "content": "\n\n---\n\n".join(entries)}


@app.get("/api/index/entries/{number}", response_model=IndexEntryResponse)
async def get_index_entry(number: int):
    """Return the study entry at position *number* (1-based) and the total entry count."""
    entries = await run_in_threadpool(read_index_entries)
    total = len(entries)
    if total == 0:
        raise HTTPException(status_code=404, detail="index.md is empty or does not exist.")
    if number < 1 or number > total:
        raise HTTPException(status_code=404, detail=f"Entry {number} not found. Index contains {total} entries.")
    return IndexEntryResponse(number=number, total=total, entry=entries[number - 1])


@app.post("/api/index", response_model=IndexResponse)
async def index_pdfs(files: Annotated[list[UploadFile], File()]):
    """Upload one or more PDFs and add them to the vector store."""
    if not files:
        raise HTTPException(status_code=422, detail="No files provided.")

    tmp_dir = tempfile.mkdtemp()
    try:
        pdf_paths: list[Path] = []
        for upload in files:
            if not (upload.filename or "").lower().endswith(".pdf"):
                raise HTTPException(status_code=422, detail=f"{upload.filename!r} is not a PDF.")
            dest = Path(tmp_dir) / upload.filename
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            pdf_paths.append(dest)

        def _run() -> int:
            return run_index(pdf_paths)

        total_chunks = await run_in_threadpool(_run)
        invalidate_rag_cache()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IndexResponse(
        indexed=[p.name for p in pdf_paths],
        total_chunks=total_chunks,
    )
