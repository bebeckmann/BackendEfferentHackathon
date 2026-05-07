"""Indexing — convert PDFs, chunk, embed, and store in Qdrant."""
from __future__ import annotations

import base64
import io
import json
import os
import textwrap
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/Besen_2016.pdf"),
]
COLLECTION = "visual_grounding"
QDRANT_PATH = "./qdrant_storage"
DOC_STORE_DIR = Path("./doc_store")
EMBED_DIM = 1536
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CONTEXT_WINDOW = 3

# ── Logging helper ────────────────────────────────────────────────────────────

def _log(tag: str, msg: str) -> None:
    print(f"  [{tag}] {msg}", flush=True)

def _log_block(tag: str, label: str, text: str, max_chars: int = 400) -> None:
    preview = text[:max_chars] + ("…" if len(text) > max_chars else "")
    indented = textwrap.indent(preview, "    │ ")
    print(f"  [{tag}] {label}:\n{indented}", flush=True)

# ── Document Converter ────────────────────────────────────────────────────────

def make_converter() -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    generate_page_images=True,
                    generate_picture_images=True,
                    images_scale=2.0,
                )
            )
        }
    )

# ── OpenRouter client ─────────────────────────────────────────────────────────

def make_openrouter_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE,
    )

# ── Document summary ──────────────────────────────────────────────────────────

def summarize_document(client: OpenAI, doc: Any) -> str:
    text = doc.export_to_text()
    model = os.getenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    _log("summarize", f"model={model}  doc_chars={len(text)}  sending={min(len(text), 18_000)}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this PDF for downstream retrieval indexing. "
                    "Focus on topic, methods, key entities, and conclusions. "
                    "Keep it under 220 words.\n\n"
                    f"{text[:18_000]}"
                ),
            }
        ],
        max_tokens=300,
    )
    summary = response.choices[0].message.content.strip()
    usage = response.usage
    _log("summarize", f"prompt_tokens={usage.prompt_tokens}  completion_tokens={usage.completion_tokens}")
    _log_block("summarize", "result", summary)
    return summary

# ── Surrounding-paragraph context ─────────────────────────────────────────────

def build_context_index(doc: Any) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Return an ordered (label, text) list and a self_ref→index mapping."""
    ordered: list[tuple[str, str]] = []
    ref_to_idx: dict[str, int] = {}
    for item, _ in doc.iterate_items():
        ref = getattr(item, "self_ref", "") or ""
        text = getattr(item, "text", "") or ""
        if not text and hasattr(item, "caption_text"):
            cap = item.caption_text(doc) if callable(item.caption_text) else item.caption_text
            text = str(cap).strip() if cap else ""
        label = getattr(getattr(item, "label", None), "value", str(getattr(item, "label", "")))
        if ref:
            ref_to_idx[ref] = len(ordered)
            ordered.append((label, text))
    _log("context-index", f"indexed {len(ordered)} elements")
    return ordered, ref_to_idx


def surrounding_paragraphs(
    ordered: list[tuple[str, str]],
    ref_to_idx: dict[str, int],
    picture_ref: str,
    window: int = CONTEXT_WINDOW,
) -> str:
    idx = ref_to_idx.get(picture_ref)
    if idx is None:
        _log("context", f"ref {picture_ref!r} not found in index — no context")
        return ""
    start = max(0, idx - window)
    end = min(len(ordered), idx + window + 1)
    snippets = [
        f"[{label}] {text}"
        for pos, (label, text) in enumerate(ordered[start:end], start=start)
        if pos != idx and text.strip()
    ]
    context = "\n".join(snippets)
    _log("context", f"ref={picture_ref!r}  neighbours={len(snippets)}  chars={len(context)}")
    return context

# ── Image description via vision LLM ─────────────────────────────────────────

def _image_to_base64(picture_item: Any, doc: Any) -> str | None:
    try:
        pil_image = picture_item.get_image(doc)
        if pil_image is None:
            _log("image", "get_image() returned None — no pixel data available")
            return None
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode()
        _log("image", f"size={pil_image.size}  png_bytes={len(raw)}  b64_chars={len(b64)}")
        return b64
    except Exception as exc:
        _log("image", f"get_image() failed: {exc}")
        return None


def describe_image(
    client: OpenAI,
    picture_item: Any,
    doc: Any,
    document_summary: str,
    context: str,
) -> str:
    image_b64 = _image_to_base64(picture_item, doc)

    caption = ""
    if hasattr(picture_item, "caption_text"):
        cap = picture_item.caption_text(doc) if callable(picture_item.caption_text) else picture_item.caption_text
        caption = str(cap).strip() if cap else ""
    _log("image", f"caption={'yes (' + caption[:60] + ')' if caption else 'none'}")

    _log_block("image", "document_summary (sent to LLM)", document_summary)
    _log_block("image", "surrounding context (sent to LLM)", context or "[none]")

    user_content: list[Any] = [
        {
            "type": "text",
            "text": (
                f"Document summary:\n{document_summary}\n\n"
                f"Surrounding paragraphs:\n{context or '[none]'}\n\n"
                f"Caption: {caption or '[none]'}\n\n"
                "Describe this image for retrieval indexing. Include what it depicts, "
                "key terms from the surrounding context, and any visible labels or data. "
                "Do not invent values not supported by the provided text."
            ),
        }
    ]
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
    else:
        _log("image", "WARNING: no image data — LLM will rely on text context only")

    model = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")
    _log("image", f"calling vision LLM  model={model}  has_image={'yes' if image_b64 else 'no'}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are preparing a PDF image element for vector retrieval. "
                    "Write a concise, retrieval-friendly description."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=400,
    )
    description = response.choices[0].message.content.strip()
    usage = response.usage
    _log("image", f"prompt_tokens={usage.prompt_tokens}  completion_tokens={usage.completion_tokens}")
    _log_block("image", "description", description)
    return description

# ── Indexing ──────────────────────────────────────────────────────────────────

def index(
    sources: list[Path],
    *,
    collection: str = COLLECTION,
    qdrant_path: str = QDRANT_PATH,
    doc_store_dir: Path = DOC_STORE_DIR,
    drop_old: bool = False,
) -> None:
    """Convert PDFs, chunk, embed with OpenRouter, and store in Qdrant.

    Images are described by a vision LLM using the document summary and the
    surrounding paragraphs as additional context.
    """
    doc_store_dir.mkdir(parents=True, exist_ok=True)
    converter = make_converter()
    chunker = HybridChunker()
    openrouter = make_openrouter_client()

    manifest: dict[str, str] = {}
    lc_docs: list[Document] = []

    for source in sources:
        print(f"\n{'─'*60}", flush=True)
        _log("convert", f"source={source}  exists={source.exists()}")
        dl_doc = converter.convert(source=str(source)).document

        page_count = len(dl_doc.pages) if hasattr(dl_doc, "pages") else "?"
        picture_count = sum(
            1 for item, _ in dl_doc.iterate_items()
            if getattr(item, "label", None) == DocItemLabel.PICTURE
        )
        _log("convert", f"pages={page_count}  pictures={picture_count}  hash={str(dl_doc.origin.binary_hash)[:12]}…")

        json_path = doc_store_dir / f"{dl_doc.origin.binary_hash}.json"
        dl_doc.save_as_json(json_path)
        _log("convert", f"doc JSON saved → {json_path}")
        manifest[dl_doc.origin.binary_hash] = str(json_path)

        print(flush=True)
        _log("summarize", f"summarizing {source.name}…")
        doc_summary = summarize_document(openrouter, dl_doc)

        print(flush=True)
        ordered, ref_to_idx = build_context_index(dl_doc)

        # self_ref → PictureItem for authoritative get_image() access
        pictures: dict[str, Any] = {
            item.self_ref: item
            for item, _ in dl_doc.iterate_items()
            if getattr(item, "label", None) == DocItemLabel.PICTURE
            and getattr(item, "self_ref", "")
        }
        _log("context-index", f"picture lookup built  entries={len(pictures)}")

        print(flush=True)
        all_chunks = list(chunker.chunk(dl_doc))
        text_chunks = sum(
            1 for c in all_chunks
            if not any(getattr(di, "label", None) == DocItemLabel.PICTURE for di in c.meta.doc_items)
        )
        image_chunks = len(all_chunks) - text_chunks
        _log("chunk", f"total={len(all_chunks)}  text={text_chunks}  image={image_chunks}")

        for i, chunk in enumerate(all_chunks):
            picture_doc_items = [
                di for di in chunk.meta.doc_items
                if getattr(di, "label", None) == DocItemLabel.PICTURE
            ]

            if picture_doc_items:
                pic_ref = getattr(picture_doc_items[0], "self_ref", "")
                pic_item = pictures.get(pic_ref, picture_doc_items[0])
                context = surrounding_paragraphs(ordered, ref_to_idx, pic_ref)
                print(flush=True)
                _log("image", f"chunk {i+1}/{len(all_chunks)}  ref={pic_ref!r}")
                page_content = describe_image(
                    client=openrouter,
                    picture_item=pic_item,
                    doc=dl_doc,
                    document_summary=doc_summary,
                    context=context,
                )
            else:
                page_content = chunk.text
                headings = chunk.meta.headings or []
                _log("chunk", f"chunk {i+1}/{len(all_chunks)}  chars={len(page_content)}  headings={headings}")

            lc_docs.append(
                Document(
                    page_content=page_content,
                    metadata={"dl_meta": chunk.meta.model_dump(mode="json")},
                )
            )

    manifest_path = doc_store_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _log("manifest", f"saved → {manifest_path}  entries={len(manifest)}")

    print(f"\n{'─'*60}", flush=True)
    _log("embed", f"total chunks to embed: {len(lc_docs)}")
    embed_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    _log("embed", f"model={embed_model}  dimensions={EMBED_DIM}")

    api_key = os.environ["OPENROUTER_API_KEY"]
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        dimensions=EMBED_DIM,
    )

    client = QdrantClient(path=qdrant_path)

    if drop_old:
        try:
            client.delete_collection(collection)
            _log("qdrant", f"dropped existing collection '{collection}'")
        except Exception:
            pass

    if collection not in {c.name for c in client.get_collections().collections}:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        _log("qdrant", f"created collection '{collection}'  size={EMBED_DIM}  distance=COSINE")
    else:
        _log("qdrant", f"collection '{collection}' already exists — appending")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )
    vector_store.add_documents(lc_docs)
    _log("qdrant", f"stored {len(lc_docs)} documents in '{collection}' at {qdrant_path}")

    print(f"\n{'─'*60}", flush=True)
    print("  Done.", flush=True)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Indexing documents…")
    index(SOURCES, drop_old=True)
