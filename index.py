"""Indexing — convert PDFs, chunk, embed, and store in Qdrant."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, ImageRefMode
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/Besen_2016.pdf"),
]
COLLECTION = "visual_grounding"
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
DOC_STORE_DIR = Path("./doc_store")
EMBED_DIM = 1536
CONTEXT_WINDOW = 3

# ── Logging helper ────────────────────────────────────────────────────────────

def _log(tag: str, msg: str) -> None:
    print(f"  [{tag}] {msg}", flush=True)

def _log_block(tag: str, label: str, text: str, max_chars: int = 400) -> None:
    preview = text[:max_chars] + ("…" if len(text) > max_chars else "")
    indented = textwrap.indent(preview, "    │ ")
    print(f"  [{tag}] {label}:\n{indented}", flush=True)

# ── Title extraction ─────────────────────────────────────────────────────────

def _extract_doc_title(dl_doc: Any, fallback: str) -> str:
    """Extract the paper title directly from the docling document structure."""
    for item, _ in dl_doc.iterate_items():
        label = getattr(item, "label", None)
        label_val = getattr(label, "value", "") if label else ""
        text = (getattr(item, "text", "") or "").strip()
        if label_val == "title" and text:
            return text
    for item, _ in dl_doc.iterate_items():
        label = getattr(item, "label", None)
        label_val = getattr(label, "value", "") if label else ""
        text = (getattr(item, "text", "") or "").strip()
        if label_val == "section_header" and len(text) > 20:
            return text
    return fallback

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

# ── OpenAI client ─────────────────────────────────────────────────────────────

def make_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

# ── Document summary ──────────────────────────────────────────────────────────

def summarize_document(client: OpenAI, doc: Any) -> str:
    text = doc.export_to_text()
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
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

# ── Paper metadata extraction ─────────────────────────────────────────────────

_METADATA_SCHEMA = (
    '{"title": "", "authors": [], "year": null, "journal": "", '
    '"doi": "", "abstract": "", "keywords": [], "language": "", "document_type": ""}'
)

def extract_paper_metadata(client: OpenAI, doc: Any) -> dict[str, Any]:
    """Extract structured metadata from document text using a JSON-mode LLM call."""
    text = doc.export_to_text()
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    _log("metadata", f"model={model}  sending={min(len(text), 8_000)}")

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract academic paper metadata and return valid JSON. "
                    f"Return exactly this schema:\n{_METADATA_SCHEMA}\n"
                    "Rules: use null for missing fields; "
                    "authors and keywords must be arrays of strings; "
                    "year must be an integer or null."
                ),
            },
            {
                "role": "user",
                "content": f"Extract metadata from this document:\n\n{text[:8_000]}",
            },
        ],
        max_tokens=500,
    )

    try:
        meta = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        _log("metadata", f"JSON parse failed: {exc}")
        meta = {}

    result: dict[str, Any] = {
        "title":         meta.get("title") or "",
        "authors":       meta.get("authors") or [],
        "year":          meta.get("year"),
        "journal":       meta.get("journal") or "",
        "doi":           meta.get("doi") or "",
        "abstract":      meta.get("abstract") or "",
        "keywords":      meta.get("keywords") or [],
        "language":      meta.get("language") or "",
        "document_type": meta.get("document_type") or "",
    }
    _log("metadata", f"title={result['title']!r}  year={result['year']}  authors={result['authors']}")
    return result


# ── Study-field extraction → index.md ────────────────────────────────────────

class StudyEntry(BaseModel):
    """One predictor/comparison evaluated in the paper."""
    study_title: str = Field(description="Full study title")
    authors: str = Field(description="Authors separated by semicolons")
    year: Optional[int] = Field(default=None, description="Publication year as integer")
    journal: str = Field(description="Journal name")
    doi: str = Field(description="DOI string, e.g. 10.xxxx/xxxxx")
    keywords: str = Field(description="Keywords separated by semicolons")
    population: str = Field(description="Study population and setting")
    sample_size: str = Field(description="Sample size, e.g. N=286 total; N=163 non-survivors; N=123 survivors")
    predictor: str = Field(description="The predictor score or variable being evaluated, including full name and abbreviation")
    outcome: str = Field(description="Primary outcome measured, e.g. 30-day mortality")
    timing: str = Field(description="When the predictor was measured relative to admission or event")
    method: str = Field(description="Statistical methods: study design, ROC analysis, tests used")
    effect_size: str = Field(description="Cutoffs, medians with p-values, or other effect size measures")
    performance: str = Field(description="AUC with CI, sensitivity, specificity, PPV, NPV, accuracy")
    notes: str = Field(description="Comparison notes or caveats relative to other predictors in the paper")
    summary: str = Field(description="One- or two-sentence clinical summary of findings for this predictor")
    source: str = Field(description="Section(s) and page numbers where data were found")


def extract_study_fields(markdown_text: str) -> StudyEntry:
    """LangChain structured-output call that extracts exactly one study entry per document."""
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    _log("study-fields", f"model={model}  md_chars={len(markdown_text)}  sending={min(len(markdown_text), 15_000)}")

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    structured_llm = llm.with_structured_output(StudyEntry)

    prompt = (
        "You are a medical-literature analyst. "
        "Extract a single summary entry for the research paper below. "
        "If the paper evaluates multiple predictors, list them all in the Predictor field "
        "and combine their performance metrics in the Performance field. "
        "Use null for missing numeric fields. Authors and keywords must be semicolon-separated strings.\n\n"
        f"---PAPER START---\n{markdown_text[:15_000]}\n---PAPER END---"
    )

    entry: StudyEntry = structured_llm.invoke(prompt)
    _log("study-fields", f"extracted entry doi={entry.doi!r}")
    return entry


def _format_study_entry(entry: StudyEntry) -> str:
    return (
        f"**Study Title:** {entry.study_title}  \n"
        f"**Authors:** {entry.authors}  \n"
        f"**Year:** {entry.year}  \n"
        f"**Journal:** {entry.journal}  \n"
        f"**Doi:** {entry.doi}  \n"
        f"**Keywords:** {entry.keywords}  \n"
        f"**Population:** {entry.population}  \n"
        f"**Sample Size:** {entry.sample_size}  \n"
        f"**Predictor:** {entry.predictor}  \n"
        f"**Outcome:** {entry.outcome}  \n"
        f"**Timing:** {entry.timing}  \n"
        f"**Method:** {entry.method}  \n"
        f"**Effect Size:** {entry.effect_size}  \n"
        f"**Performance:** {entry.performance}  \n"
        f"**Notes:** {entry.notes}  \n"
        f"**Summary:** {entry.summary}  \n"
        f"**Source:** {entry.source}  \n"
        "\n---\n\n"
    )


def _existing_dois(index_path: Path) -> set[str]:
    """Return DOIs already recorded in index.md."""
    if not index_path.exists():
        return set()
    return {
        line.removeprefix("**Doi:**").strip().rstrip("  ")
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("**Doi:**")
    }


def append_to_index_md(entry: StudyEntry, index_path: Path) -> None:
    """Append one study entry to index.md; skip if DOI already present."""
    if not index_path.exists():
        index_path.write_text("# Study Index\n\n", encoding="utf-8")

    if entry.doi and entry.doi in _existing_dois(index_path):
        _log("index-md", f"doi={entry.doi!r} already present — skipping")
        return

    with index_path.open("a", encoding="utf-8") as f:
        f.write(_format_study_entry(entry))
    _log("index-md", f"appended entry doi={entry.doi!r} → {index_path}")


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

    user_content: list[Any] = []
    if caption:
        user_content.append({"type": "text", "text": f"Caption: {caption}\n\nDescribe only what is visually shown in this image."})
    else:
        user_content.append({"type": "text", "text": "Describe only what is visually shown in this image."})

    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
    else:
        _log("image", "WARNING: no image data — LLM will rely on text context only")

    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    _log("image", f"calling vision LLM  model={model}  has_image={'yes' if image_b64 else 'no'}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You describe images exactly as they appear visually. "
                    "Do not reference any surrounding text, paper context, or external knowledge. "
                    "Only describe what is directly visible in the image."
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
    doc_store_dir: Path = DOC_STORE_DIR,
    drop_old: bool = False,
) -> int:
    """Convert PDFs, chunk, embed with OpenAI, and store in Qdrant.

    Images are described by a vision LLM using the document summary and the
    surrounding paragraphs as additional context.
    """
    doc_store_dir.mkdir(parents=True, exist_ok=True)
    converter = make_converter()
    openai_client = make_openai_client()

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

        doc_hash = str(dl_doc.origin.binary_hash)
        json_path = doc_store_dir / f"{doc_hash}.json"
        dl_doc.save_as_json(json_path, image_mode=ImageRefMode.EMBEDDED)
        _log("convert", f"doc JSON saved → {json_path}")
        manifest[doc_hash] = str(json_path)

        print(flush=True)
        _log("summarize", f"summarizing {source.name}…")
        doc_summary = summarize_document(openai_client, dl_doc)

        print(flush=True)
        _log("metadata", f"extracting paper metadata from {source.name}…")
        paper_meta = extract_paper_metadata(openai_client, dl_doc)

        doc_title = _extract_doc_title(dl_doc, doc_hash)
        safe_title = re.sub(r'[\\/:*?"<>|]', '', doc_title).strip()
        md_path = doc_store_dir / f"{safe_title}.md"

        print(flush=True)
        ordered, ref_to_idx = build_context_index(dl_doc)

        print(flush=True)
        current_heading: str | None = None
        text_count = image_count = 0
        image_descriptions: list[str] = []

        for item, _ in dl_doc.iterate_items():
            label = getattr(item, "label", None)
            prov_list = getattr(item, "prov", []) or []
            page_nos = sorted({p.page_no for p in prov_list})

            if label == DocItemLabel.SECTION_HEADER:
                current_heading = getattr(item, "text", "") or ""

            if label == DocItemLabel.PICTURE:
                pic_ref = getattr(item, "self_ref", "")
                context = surrounding_paragraphs(ordered, ref_to_idx, pic_ref)
                print(flush=True)
                _log("image", f"ref={pic_ref!r}  pages={page_nos}")
                page_content = describe_image(
                    client=openai_client,
                    picture_item=item,
                    doc=dl_doc,
                    document_summary=doc_summary,
                    context=context,
                )
                image_descriptions.append(page_content)
                image_count += 1
            else:
                text = getattr(item, "text", "") or ""
                if not text.strip():
                    continue
                page_content = text
                _log("chunk", f"label={label}  chars={len(page_content)}  heading={current_heading!r}  pages={page_nos}")
                text_count += 1

            lc_docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        # paper-level fields
                        "paper_title":         paper_meta["title"],
                        "paper_authors":       paper_meta["authors"],
                        "paper_year":          paper_meta["year"],
                        "paper_journal":       paper_meta["journal"],
                        "paper_doi":           paper_meta["doi"],
                        "paper_abstract":      paper_meta["abstract"],
                        "paper_keywords":      paper_meta["keywords"],
                        "paper_language":      paper_meta["language"],
                        "paper_document_type": paper_meta["document_type"],
                        # chunk-level fields
                        "label":    label.value if hasattr(label, "value") else str(label),
                        "heading":  current_heading,
                        "page_nos": page_nos,
                        "self_ref": getattr(item, "self_ref", ""),
                        "doc_hash": str(dl_doc.origin.binary_hash),
                        "md_file":  md_path.name,
                    },
                )
            )

        _log("chunk", f"total={text_count + image_count}  text={text_count}  image={image_count}")

        md_text = dl_doc.export_to_markdown()
        for desc in image_descriptions:
            md_text = md_text.replace("<!-- image -->", f"> {desc}\n", 1)
        md_path.write_text(md_text, encoding="utf-8")
        _log("convert", f"doc Markdown saved → {md_path}  images_injected={len(image_descriptions)}")

        print(flush=True)
        _log("study-fields", f"extracting structured study fields from {md_path.name}…")
        study_entry = extract_study_fields(md_text)
        index_md_path = doc_store_dir / "index.md"
        append_to_index_md(study_entry, index_md_path)

    manifest_path = doc_store_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing.update(manifest)
        manifest = existing
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _log("manifest", f"saved → {manifest_path}  entries={len(manifest)}")

    print(f"\n{'─'*60}", flush=True)
    _log("embed", f"total chunks to embed: {len(lc_docs)}")
    embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    _log("embed", f"model={embed_model}  dimensions={EMBED_DIM}")

    embeddings = OpenAIEmbeddings(
        model=embed_model,
        api_key=os.environ["OPENAI_API_KEY"],
        dimensions=EMBED_DIM,
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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
    _log("qdrant", f"stored {len(lc_docs)} documents in '{collection}' at {QDRANT_URL}")

    print(f"\n{'─'*60}", flush=True)
    print("  Done.", flush=True)
    return len(lc_docs)


# ── Index-md reader ──────────────────────────────────────────────────────────

def read_index_entries(index_path: Path = DOC_STORE_DIR / "index.md") -> list[str]:
    """Return each study entry from index.md as a markdown string (1 per PDF)."""
    if not index_path.exists():
        return []
    content = index_path.read_text(encoding="utf-8")
    content = re.sub(r"^#[^\n]*\n+", "", content)  # strip header line
    entries = [e.strip() for e in content.split("\n---\n")]
    return [e for e in entries if e]


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Indexing documents…")
    index(SOURCES, drop_old=True)
