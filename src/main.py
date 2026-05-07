import base64
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.agent import ask_agent
from src.schemas import AgentResponse, EvidenceImage, HealthResponse
from src.storage import save_uploaded_images

load_dotenv()

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "http://localhost:8000"
)

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,https://www.sepsis-analysis.online,https://sepsis-analysis.online"
).split(",")

FRONTEND_ORIGINS = [origin.strip() for origin in FRONTEND_ORIGINS if origin.strip()]


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
EVIDENCE_DIR = STATIC_DIR / "evidence"

app = FastAPI(
    title="Sepsis Research Agent API",
    version="0.1.0",
)

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
    Accepts:
    - text message
    - optional image files

    Returns:
    - agent answer
    - evidence image URLs that Next.js can render alongside the text
    """
    uploaded_images = await save_uploaded_images(
        files=images or [],
        evidence_dir=EVIDENCE_DIR,
        public_base_url=PUBLIC_BASE_URL,
    )

    answer = await ask_agent(
        message=message,
        uploaded_images=uploaded_images,
    )

    evidence = [
        EvidenceImage(
            url=image["url"],
            caption=f"Uploaded evidence image: {image['filename']}",
            source=image["filename"],
            kind="uploaded",
        )
        for image in uploaded_images
    ]

    return AgentResponse(
        answer=answer,
        images=evidence,
        warnings=[
            "Research-use only. Not for clinical diagnosis, triage, or treatment decisions."
        ],
    )

import base64
from pathlib import Path


@app.get("/api/success-image")
async def success_image_base64():
    image_files = [
        STATIC_DIR / "data" / "success.png",
        STATIC_DIR / "data" / "success-2.png",
    ]

    images = []

    for image_path in image_files:
        if not image_path.exists():
            return {
                "error": f"{image_path.name} not found",
                "path": str(image_path),
            }

        image_bytes = image_path.read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        images.append(
            {
                "filename": image_path.name,
                "mime_type": "image/png",
                "base64": image_base64,
                "data_url": f"data:image/png;base64,{image_base64}",
            }
        )

    return {
        "images": images
    }