"""Retrieval — load Qdrant index, run RAG queries with visual grounding."""
from __future__ import annotations

import io
import json
import os
import re
import threading
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import base64 as _base64

from PIL import Image, ImageDraw, ImageFont
from docling.datamodel.document import DoclingDocument
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchValue, VectorParams

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION = "visual_grounding"
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
DOC_STORE_DIR = Path("./doc_store")
EMBED_DIM = 1536
TOP_K = 3

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer using only the retrieved context below. "
        "Cite every statement with the source number in square brackets, e.g. [1], [2]. "
        "Every statement must have at least one citation. "
        "If the answer is not found in the context, say so.\n\nContext:\n{context}",
    ),
    ("human", "{input}"),
])

# ── Numbered chunk dataclass ──────────────────────────────────────────────────

@dataclass
class CitedChunk:
    num: int          # sequential citation number used in the answer as [N]
    pdf_name: str     # PDF filename (basename)
    page_no: int      # earliest page the chunk appears on
    doc_hash: str     # Docling binary hash for doc_store lookup
    lc_doc: Document  # original LangChain document
    is_image: bool    # True when chunk represents a figure/picture
    docling_id: str   # self_ref of first doc_item (e.g. "#/texts/3")


# ── Load persisted doc store ──────────────────────────────────────────────────

def load_doc_store(doc_store_dir: Path = DOC_STORE_DIR) -> dict[str, Path]:
    """Load the binary_hash → json_path mapping written by index.py."""
    manifest_path = doc_store_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    raw = json.loads(manifest_path.read_text())
    return {k: Path(v) for k, v in raw.items()}


# ── Sorting and numbering ─────────────────────────────────────────────────────

def _sort_and_number_chunks(docs: list[Document]) -> list[CitedChunk]:
    """Sort retrieved docs by (pdf_name, page_no) and assign citation numbers."""
    def _key(doc: Document) -> tuple[str, int]:
        m = doc.metadata
        name = m.get("paper_title") or m.get("doc_hash", "")
        return (name, min(m.get("page_nos", [0]), default=0))

    result: list[CitedChunk] = []
    for i, doc in enumerate(sorted(docs, key=_key), start=1):
        m = doc.metadata
        result.append(CitedChunk(
            num=i,
            pdf_name=m.get("paper_title") or m.get("doc_hash", "unknown"),
            page_no=min(m.get("page_nos", [0]), default=0),
            doc_hash=str(m.get("doc_hash", "")),
            lc_doc=doc,
            is_image=m.get("label") == "picture",
            docling_id=m.get("self_ref", "unknown"),
        ))
    return result


# ── Context markdown builder ──────────────────────────────────────────────────

def _paper_meta_block(meta: dict) -> str:
    """Format paper-level metadata as a markdown definition list."""
    fields: list[tuple[str, str]] = [
        ("Title",         meta.get("paper_title", "")),
        ("Authors",       ", ".join(meta.get("paper_authors", []) or [])),
        ("Year",          str(meta.get("paper_year", "")) if meta.get("paper_year") else ""),
        ("Journal",       meta.get("paper_journal", "")),
        ("DOI",           meta.get("paper_doi", "")),
        ("Language",      meta.get("paper_language", "")),
        ("Document type", meta.get("paper_document_type", "")),
        ("Keywords",      ", ".join(meta.get("paper_keywords", []) or [])),
        ("Abstract",      meta.get("paper_abstract", "")),
    ]
    rows = "\n".join(f"**{k}:** {v}" for k, v in fields if v)
    return rows


def _build_context_markdown(cited_chunks: list[CitedChunk]) -> str:
    """Build structured markdown context for the LLM with citation numbers."""
    lines: list[str] = []

    # Group: pdf_name → page_no → chunks (order preserved because list is sorted)
    by_pdf: dict[str, dict[int, list[CitedChunk]]] = {}
    first_chunk_per_pdf: dict[str, CitedChunk] = {}
    for cc in cited_chunks:
        by_pdf.setdefault(cc.pdf_name, {}).setdefault(cc.page_no, []).append(cc)
        first_chunk_per_pdf.setdefault(cc.pdf_name, cc)

    for pdf_name, pages in by_pdf.items():
        lines.append(f"# {pdf_name}\n")
        meta_block = _paper_meta_block(first_chunk_per_pdf[pdf_name].lc_doc.metadata)
        if meta_block:
            lines.append(meta_block + "\n")
        for page_no in sorted(pages):
            lines.append(f"## page {page_no}\n")
            for cc in pages[page_no]:
                lines.append(f"### [{cc.num}] chunk {cc.docling_id}\n")
                if cc.is_image:
                    lines.append(f"*Image description:* {cc.lc_doc.page_content}\n")
                else:
                    lines.append(f"{cc.lc_doc.page_content}\n")

    return "\n".join(lines)


# ── Citation extraction ───────────────────────────────────────────────────────

def _extract_cited_nums(answer: str) -> set[int]:
    """Return the set of [N] citation numbers referenced in the answer."""
    return {int(m) for m in re.findall(r"\[(\d+)\]", answer)}


# ── RAG chain ─────────────────────────────────────────────────────────────────

def build_chain(vector_store: QdrantVectorStore) -> tuple:
    """Return (retrieval_runnable, llm) for use in ask()."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieval_chain = RunnableParallel(
        context=itemgetter("input") | retriever,
        input=itemgetter("input"),
    )
    return retrieval_chain, llm


# ── Visual grounding ──────────────────────────────────────────────────────────

def _pil_from_page_image(page_image) -> Image.Image | None:
    """Return a PIL image from a Docling PageItem.image, decoding base64 if needed."""
    if page_image is None:
        return None
    if getattr(page_image, "pil_image", None) is not None:
        return page_image.pil_image
    uri_obj = getattr(page_image, "uri", None)
    if uri_obj is None:
        return None
    uri = str(uri_obj)
    print(f"  [grounding] page_image.uri type={type(uri_obj).__name__}  starts_with_data={uri.startswith('data:')}")
    if uri.startswith("data:"):
        try:
            _, b64_data = uri.split(",", 1)
            return Image.open(io.BytesIO(_base64.b64decode(b64_data)))
        except Exception as exc:
            print(f"  [grounding] base64 decode failed: {exc}")
    return None


def _load_page_image_from_json(json_path: Path, page_no: int) -> Image.Image | None:
    """Directly parse the Docling JSON to extract the embedded page image.

    This bypasses Docling/Pydantic deserialization, which may silently drop
    data: URIs during validation.
    """
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        pages = raw.get("pages", {})
        page_data = pages.get(str(page_no)) or pages.get(page_no)
        if not page_data:
            print(f"  [grounding] page_no={page_no} not found in JSON keys: {list(pages.keys())[:5]}")
            return None
        image_data = page_data.get("image") or {}
        uri = image_data.get("uri", "") if isinstance(image_data, dict) else ""
        if not uri:
            print(f"  [grounding] no uri in page image JSON")
            return None
        print(f"  [grounding] raw JSON uri prefix: {uri[:60]}")
        if uri.startswith("data:"):
            _, b64 = uri.split(",", 1)
            return Image.open(io.BytesIO(_base64.b64decode(b64)))
    except Exception as exc:
        print(f"  [grounding] _load_page_image_from_json failed: {exc}")
    return None


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow ≥ 10
    except TypeError:
        return ImageFont.load_default()


def highlight_sources(
    cited_chunks: list[CitedChunk],
    cited_nums: set[int],
    doc_store: dict[str, Path],
    *,
    num_labels: dict[int, int] | None = None,
    color: str = "blue",
    label_bg: str = "blue",
    label_fg: str = "white",
    line_width: int = 2,
) -> list[Image.Image]:
    """Draw bounding boxes only for cited chunks; label each box with [N].

    Returns one annotated PIL image per (document, page) that contains at
    least one cited source.
    """
    active = [cc for cc in cited_chunks if cc.num in cited_nums]
    print(f"  [grounding] active chunks: {len(active)}  cited_nums={sorted(cited_nums)}")

    dl_docs: dict[str, DoclingDocument] = {}

    # (doc_hash, page_no) → list of (citation_num, prov)
    # Look up bounding boxes by finding the item via self_ref in the docling document.
    page_entries: dict[tuple[str, int], list[tuple[int, object]]] = {}
    for cc in active:
        print(f"  [grounding] cc.num={cc.num}  doc_hash={cc.doc_hash!r}  self_ref={cc.docling_id!r}  in_store={cc.doc_hash in doc_store}")
        if not cc.doc_hash or cc.doc_hash not in doc_store:
            continue
        if cc.doc_hash not in dl_docs:
            dl_docs[cc.doc_hash] = DoclingDocument.load_from_json(doc_store[cc.doc_hash])
        dl_doc = dl_docs[cc.doc_hash]
        self_ref = cc.lc_doc.metadata.get("self_ref", "")
        matched = False
        for item, _ in dl_doc.iterate_items():
            if getattr(item, "self_ref", "") == self_ref:
                matched = True
                prov_list = getattr(item, "prov", []) or []
                print(f"  [grounding]   matched item  prov_count={len(prov_list)}")
                for prov in prov_list:
                    key = (cc.doc_hash, prov.page_no)
                    page_entries.setdefault(key, []).append((cc.num, prov))
                break
        if not matched:
            print(f"  [grounding]   no item matched self_ref={self_ref!r}")

    print(f"  [grounding] page_entries keys: {list(page_entries.keys())}")
    font = _load_font(18)
    annotated: list[Image.Image] = []

    for (doc_hash, page_no), num_provs in page_entries.items():
        try:
            if doc_hash not in doc_store:
                print(f"  [grounding] doc_hash {doc_hash!r} NOT in doc_store — skipping")
                continue
            if doc_hash not in dl_docs:
                dl_docs[doc_hash] = DoclingDocument.load_from_json(doc_store[doc_hash])

            pages_keys = list(dl_docs[doc_hash].pages.keys())
            print(f"  [grounding] pages keys={pages_keys}  looking for page_no={page_no} (type={type(page_no).__name__})")
            page = dl_docs[doc_hash].pages[page_no]
            print(f"  [grounding] page.image={page.image!r}")
            pil_img = _pil_from_page_image(page.image)
            if pil_img is None:
                print(f"  [grounding] pil_from_page_image returned None, trying raw JSON fallback")
                pil_img = _load_page_image_from_json(doc_store[doc_hash], page_no)
            if pil_img is None:
                print(f"  [grounding] no image data for page {page_no} in {doc_hash}")
                continue
        except Exception as exc:
            import traceback
            print(f"  [grounding] EXCEPTION in rendering loop: {exc}")
            traceback.print_exc()
            continue
        img = pil_img.copy()
        draw = ImageDraw.Draw(img)
        padding = line_width + 2

        for citation_num, prov in num_provs:
            bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)
            bbox = bbox.normalized(page.size)
            bbox.l = round(bbox.l * img.width - padding)
            bbox.r = round(bbox.r * img.width + padding)
            bbox.t = round(bbox.t * img.height - padding)
            bbox.b = round(bbox.b * img.height + padding)
            x0, y0 = bbox.l, bbox.t

            draw.rectangle(xy=bbox.as_tuple(), outline=color, width=line_width)

            # Draw citation label badge above the top-left corner
            display_num = num_labels[citation_num] if num_labels and citation_num in num_labels else citation_num
            label = f"[{display_num}]"
            try:
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except AttributeError:
                tw, th = font.getsize(label)

            pad = 2
            badge_y0 = max(0, y0 - th - 2 * pad)
            badge_y1 = badge_y0 + th + 2 * pad
            draw.rectangle([(x0, badge_y0), (x0 + tw + 2 * pad, badge_y1)], fill=label_bg)
            draw.text((x0 + pad, badge_y0 + pad), label, fill=label_fg, font=font)

        annotated.append(img)

    return annotated


# ── Query ─────────────────────────────────────────────────────────────────────

def ask(
    question: str,
    chain: tuple,
    doc_store: dict[str, Path],
    *,
    output_dir: Path | None = None,
) -> tuple[str, list[Image.Image]]:
    """Run structured RAG: sort chunks → markdown context → cited answer → grounding.

    Returns (answer, grounding_images). If output_dir is given the report and
    images are also saved to disk.
    """
    retrieval_chain, llm = chain

    # 1. Retrieve context documents
    resp = retrieval_chain.invoke({"input": question})
    context_docs: list[Document] = resp["context"]

    # 2. Sort by (pdf, page) and assign [N] citation numbers
    cited_chunks = _sort_and_number_chunks(context_docs)

    # 3. Build structured markdown context
    context_md = _build_context_markdown(cited_chunks)
    print("\n── Context sent to LLM ──────────────────────────────────────────")
    print(context_md)
    print("─────────────────────────────────────────────────────────────────\n")

    # 4. Generate answer — LLM must cite [N] for every statement
    messages = PROMPT.format_messages(context=context_md, input=question)
    answer: str = llm.invoke(messages).content

    # 5. Renumber citations [N] → [1], [2], ... in order of first appearance
    cited_in_order = list(dict.fromkeys(int(m) for m in re.findall(r"\[(\d+)\]", answer)))
    remap = {old: new for new, old in enumerate(cited_in_order, start=1)}
    answer = re.sub(r"\[(\d+)\]", lambda m: f"[{remap.get(int(m.group(1)), int(m.group(1)))}]", answer)
    original_cited_nums = set(cited_in_order)
    print(f"  Citations used in answer: {sorted(remap.values())}")

    # 6. Render grounding images
    images = highlight_sources(cited_chunks, original_cited_nums, doc_store, num_labels=remap)

    # 7. Optionally persist to disk
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            out = output_dir / f"grounding_{i}.png"
            img.save(out)
            print(f"  Grounding image saved: {out}")

    return answer, images, cited_chunks


# ── LangChain tool ───────────────────────────────────────────────────────────

from langchain_core.tools import tool  # noqa: E402

_rag_cache: dict = {}
_rag_lock = threading.Lock()

# Images produced by the most recent search_literature call.
# Using a Queue avoids thread-ID mismatches: the LangGraph agent may execute
# tools in a different thread (or async event loop) than the caller of
# pop_last_images(), so keying by threading.get_ident() is unreliable.
import queue as _queue
_images_queue: _queue.Queue = _queue.Queue()


def pop_last_images() -> list:
    """Retrieve and clear images from the most recent search_literature call."""
    try:
        return _images_queue.get_nowait()
    except _queue.Empty:
        return []


def list_indexed_documents() -> list[dict]:
    """Return [{doc_hash, paper_title}] for every entry in the manifest."""
    manifest_path = DOC_STORE_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest: dict[str, str] = json.loads(manifest_path.read_text())
    if not manifest:
        return []

    titles: dict[str, str] = {}
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        collections = {c.name for c in client.get_collections().collections}
        if COLLECTION in collections:
            offset = None
            while True:
                results, offset = client.scroll(
                    collection_name=COLLECTION,
                    with_payload=True,
                    limit=100,
                    offset=offset,
                )
                for point in results:
                    meta = (point.payload or {}).get("metadata", {})
                    dh = meta.get("doc_hash", "")
                    title = meta.get("paper_title", "")
                    if dh and dh not in titles:
                        titles[dh] = title
                if offset is None:
                    break
    finally:
        client.close()

    return [
        {"doc_hash": dh, "paper_title": titles.get(dh, "")}
        for dh in manifest
    ]


def find_doc_hash_by_title(title: str) -> str | None:
    """Return the doc_hash whose paper_title matches (case-insensitive), or None."""
    title_lower = title.lower()
    for doc in list_indexed_documents():
        if doc["paper_title"].lower() == title_lower:
            return doc["doc_hash"]
    return None


def delete_document(doc_hash: str) -> bool:
    """Delete all Qdrant chunks, the doc-store JSON, and the manifest entry for doc_hash.

    Returns True when the document existed and was removed, False when not found.
    """
    manifest_path = DOC_STORE_DIR / "manifest.json"
    if not manifest_path.exists():
        return False

    manifest: dict[str, str] = json.loads(manifest_path.read_text())
    if doc_hash not in manifest:
        return False

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        collections = {c.name for c in client.get_collections().collections}
        if COLLECTION in collections:
            client.delete(
                collection_name=COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="metadata.doc_hash", match=MatchValue(value=doc_hash))]
                    )
                ),
            )
    finally:
        client.close()

    json_path = Path(manifest[doc_hash])
    if json_path.exists():
        json_path.unlink()
    for md_path in json_path.parent.glob("*.md"):
        # remove any .md whose stem matches the json stem (uuid fallback) or
        # that was written for this document (we can't know the title here, so
        # remove all .md files that share the same parent and doc_hash stem)
        if md_path.stem == json_path.stem:
            md_path.unlink()

    del manifest[doc_hash]
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return True


def delete_all_documents() -> int:
    """Delete every document: drop and recreate the Qdrant collection, remove all
    doc-store JSON files, and reset the manifest.

    Returns the number of documents that were removed.
    """
    manifest_path = DOC_STORE_DIR / "manifest.json"
    manifest: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    count = len(manifest)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        existing = {c.name for c in client.get_collections().collections}
        if COLLECTION in existing:
            client.delete_collection(COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    finally:
        client.close()

    for json_path_str in manifest.values():
        json_path = Path(json_path_str)
        if json_path.exists():
            json_path.unlink()

    for md_path in DOC_STORE_DIR.glob("*.md"):
        md_path.unlink()

    manifest_path.write_text(json.dumps({}, indent=2))

    return count


def invalidate_rag_cache() -> None:
    """Clear the RAG cache so the next call to _get_rag() reloads from disk."""
    with _rag_lock:
        _rag_cache.clear()


def _get_rag() -> tuple:
    """Lazily initialise and cache the RAG chain + doc store."""
    if _rag_cache:
        return _rag_cache["chain"], _rag_cache["doc_store"]
    with _rag_lock:
        if _rag_cache:
            return _rag_cache["chain"], _rag_cache["doc_store"]
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.environ["OPENAI_API_KEY"],
            dimensions=EMBED_DIM,
        )
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        existing = {c.name for c in qdrant.get_collections().collections}
        if COLLECTION not in existing:
            qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        vector_store = QdrantVectorStore(
            client=qdrant,
            collection_name=COLLECTION,
            embedding=embeddings,
        )
        _rag_cache["chain"] = build_chain(vector_store)
        _rag_cache["doc_store"] = load_doc_store()
        return _rag_cache["chain"], _rag_cache["doc_store"]


@tool
def search_literature(question: str) -> str:
    """Search the indexed scientific literature and return a cited markdown answer.

    Use this tool whenever the question requires looking up facts, findings,
    methods, or conclusions from the ingested research papers.
    Input should be a plain-language question.
    """
    chain, doc_store = _get_rag()
    answer, images, cited_chunks = ask(question, chain, doc_store)

    # Store images so the API layer can retrieve them after the agent finishes.
    _images_queue.put(list(images))

    cited_nums = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    sources = "\n".join(
        f"**[{cc.num}]** {cc.pdf_name}"
        + (f" ({cc.lc_doc.metadata.get('paper_year')})" if cc.lc_doc.metadata.get("paper_year") else "")
        + f" — page {cc.page_no}"
        for cc in cited_chunks
        if cc.num in cited_nums
    )
    return f"{answer}\n\n---\n**Sources**\n\n{sources}"


@tool
def get_paper_markdown(paper_name: str) -> str:
    """Return the entire content of a paper as markdown text, given the paper title or a close approximation.

    Use this tool if you want to go deep on a specific paper, read its full content, or extract information that may not have been included in the retrieved chunks. Input should be the exact paper title or a close approximation to it.

    Input should be the paper title or a close approximation.
    """
    md_files = list(DOC_STORE_DIR.glob("*.md"))
    if not md_files:
        return "No markdown files found in the document store."

    needle = paper_name.lower()
    best: Path | None = None
    best_score = -1
    for path in md_files:
        stem = path.stem.replace("_", " ").lower()
        # score by how many words from the query appear in the filename
        score = sum(1 for word in needle.split() if word in stem)
        if score > best_score:
            best_score, best = score, path

    if best is None or best_score == 0:
        available = ", ".join(p.stem.replace("_", " ") for p in md_files)
        return f"No matching paper found for '{paper_name}'. Available papers: {available}"

    return best.read_text(encoding="utf-8")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"],
        dimensions=EMBED_DIM,
    )
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

    doc_store = load_doc_store()
    chain = build_chain(vector_store)

    try:
        question = "Why was it important to test Sepsis-3 definitions outside high-income countries?"
        print(f"\nQ: {question}")
        answer, _, _chunks = ask(question, chain, doc_store, output_dir=Path("grounding_output"))
        print(f"\nA: {answer}")
    finally:
        client.close()
