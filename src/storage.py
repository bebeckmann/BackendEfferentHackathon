import base64
import mimetypes
import uuid
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile, HTTPException

ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB per image


async def save_uploaded_images(
    files: Iterable[UploadFile],
    evidence_dir: Path,
    public_base_url: str,
) -> list[dict]:
    evidence_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict] = []

    for file in files:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type: {file.content_type}",
            )

        raw = await file.read()

        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large: {file.filename}",
            )

        ext = mimetypes.guess_extension(file.content_type or "") or ".bin"
        safe_name = f"{uuid.uuid4().hex}{ext}"
        path = evidence_dir / safe_name
        path.write_bytes(raw)

        data_url = (
            f"data:{file.content_type};base64,"
            f"{base64.b64encode(raw).decode('utf-8')}"
        )

        saved.append(
            {
                "filename": file.filename,
                "mime_type": file.content_type,
                "path": str(path),
                "url": f"{public_base_url}/static/evidence/{safe_name}",
                "data_url": data_url,
            }
        )

    return saved