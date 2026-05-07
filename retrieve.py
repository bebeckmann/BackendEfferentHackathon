"""Retrieval — load Qdrant index, run RAG queries with visual grounding."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docling.chunking import DocMeta
from docling.datamodel.document import DoclingDocument
from docling_core.types.doc import DocItemLabel
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION = "visual_grounding"
QDRANT_PATH = "./qdrant_storage"
DOC_STORE_DIR = Path("./doc_store")
EMBED_DIM = 1536
TOP_K = 3
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

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
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Run index.py first."
        )
    raw = json.loads(manifest_path.read_text())
    return {k: Path(v) for k, v in raw.items()}


# ── Sorting and numbering ─────────────────────────────────────────────────────

def _sort_and_number_chunks(docs: list[Document]) -> list[CitedChunk]:
    """Sort retrieved docs by (pdf_name, page_no) and assign citation numbers."""
    def _key(doc: Document) -> tuple[str, int]:
        meta = DocMeta.model_validate(doc.metadata["dl_meta"])
        name = Path(meta.origin.filename or str(meta.origin.binary_hash)).name
        pages = [p.page_no for di in meta.doc_items for p in (di.prov or [])]
        return (name, min(pages, default=0))

    result: list[CitedChunk] = []
    for i, doc in enumerate(sorted(docs, key=_key), start=1):
        meta = DocMeta.model_validate(doc.metadata["dl_meta"])
        pdf_name = Path(meta.origin.filename or str(meta.origin.binary_hash)).name
        pages = [p.page_no for di in meta.doc_items for p in (di.prov or [])]
        result.append(CitedChunk(
            num=i,
            pdf_name=pdf_name,
            page_no=min(pages, default=0),
            doc_hash=str(meta.origin.binary_hash),
            lc_doc=doc,
            is_image=any(
                getattr(di, "label", None) == DocItemLabel.PICTURE
                for di in meta.doc_items
            ),
            docling_id=meta.doc_items[0].self_ref if meta.doc_items else "unknown",
        ))
    return result


# ── Context markdown builder ──────────────────────────────────────────────────

def _build_context_markdown(cited_chunks: list[CitedChunk]) -> str:
    """Build structured markdown context for the LLM with citation numbers."""
    lines: list[str] = []

    # Group: pdf_name → page_no → chunks (order preserved because list is sorted)
    by_pdf: dict[str, dict[int, list[CitedChunk]]] = {}
    for cc in cited_chunks:
        by_pdf.setdefault(cc.pdf_name, {}).setdefault(cc.page_no, []).append(cc)

    for pdf_name, pages in by_pdf.items():
        lines.append(f"# {pdf_name}\n")
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
    api_key = os.environ["OPENROUTER_API_KEY"]
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini"),
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        temperature=0,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieval_chain = RunnableParallel(
        context=itemgetter("input") | retriever,
        input=itemgetter("input"),
    )
    return retrieval_chain, llm


# ── Visual grounding ──────────────────────────────────────────────────────────

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

    # (doc_hash, page_no) → list of (citation_num, prov)
    page_entries: dict[tuple[str, int], list[tuple[int, object]]] = {}
    for cc in active:
        meta = DocMeta.model_validate(cc.lc_doc.metadata["dl_meta"])
        for doc_item in meta.doc_items:
            for prov in doc_item.prov or []:
                key = (cc.doc_hash, prov.page_no)
                page_entries.setdefault(key, []).append((cc.num, prov))

    dl_docs: dict[str, DoclingDocument] = {}
    font = _load_font(18)
    annotated: list[Image.Image] = []

    for (doc_hash, page_no), num_provs in page_entries.items():
        if doc_hash not in doc_store:
            continue
        if doc_hash not in dl_docs:
            dl_docs[doc_hash] = DoclingDocument.load_from_json(doc_store[doc_hash])

        page = dl_docs[doc_hash].pages[page_no]
        img = page.image.pil_image.copy()
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
            label = f"[{citation_num}]"
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
    output_dir: Path = Path("."),
) -> str:
    """Run structured RAG: sort chunks → markdown context → cited answer → grounding."""
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

    # 5. Extract which citation numbers were actually used
    cited_nums = _extract_cited_nums(answer)
    print(f"  Citations used in answer: {sorted(cited_nums)}")

    # 6. Render bounding boxes only for cited chunks, labelled with [N]
    output_dir.mkdir(parents=True, exist_ok=True)
    images = highlight_sources(cited_chunks, cited_nums, doc_store)
    for i, img in enumerate(images):
        out = output_dir / f"grounding_{i}.png"
        img.save(out)
        print(f"  Grounding image saved: {out}")

    return answer


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ["OPENROUTER_API_KEY"]
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        dimensions=EMBED_DIM,
    )
    client = QdrantClient(path=QDRANT_PATH)
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
        answer = ask(question, chain, doc_store, output_dir=Path("notebook/grounding_output"))
        print(f"\nA: {answer}")
    finally:
        client.close()
