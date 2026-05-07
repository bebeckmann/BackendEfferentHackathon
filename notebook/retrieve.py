"""Retrieval — load Qdrant index, run RAG queries with visual grounding."""
from __future__ import annotations

import json
import os
from operator import itemgetter
from pathlib import Path

from PIL import Image, ImageDraw
from docling.chunking import DocMeta
from docling.datamodel.document import DoclingDocument
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
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
        "Answer using only the retrieved context. "
        "If the answer is not in the context, say so.\n\nContext:\n{context}",
    ),
    ("human", "{input}"),
])

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


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_chain(vector_store: QdrantVectorStore):
    """Build a LCEL retrieval-augmented generation chain.

    Returns a chain whose .invoke(question) produces:
      {"context": [Document, ...], "input": question, "answer": str}
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini"),
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        temperature=0,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    return (
        RunnableParallel(
            context=itemgetter("input") | retriever,
            input=itemgetter("input"),
        )
        .assign(
            answer=(
                {
                    "context": lambda x: _format_docs(x["context"]),
                    "input": itemgetter("input"),
                }
                | PROMPT
                | llm
                | StrOutputParser()
            )
        )
    )


# ── Visual Grounding ──────────────────────────────────────────────────────────

def highlight_sources(
    context_docs: list[Document],
    doc_store: dict[str, Path],
    *,
    color: str = "blue",
    line_width: int = 2,
) -> list[Image.Image]:
    """Draw bounding boxes on page images for every retrieved chunk.

    Chunks from the same document page are all drawn on the same image.
    """
    # Group provenance objects by (doc_hash, page_no) so same-page chunks
    # share one annotated image.
    page_entries: dict[tuple[str, int], list] = {}
    for lc_doc in context_docs:
        meta = DocMeta.model_validate(lc_doc.metadata["dl_meta"])
        for doc_item in meta.doc_items:
            for prov in doc_item.prov or []:
                key = (meta.origin.binary_hash, prov.page_no)
                page_entries.setdefault(key, []).append(prov)

    # Cache loaded documents so each JSON file is read at most once.
    dl_docs: dict[str, DoclingDocument] = {}

    annotated: list[Image.Image] = []
    for (doc_hash, page_no), provs in page_entries.items():
        if str(doc_hash) not in doc_store:
            continue
        doc_hash = str(doc_hash)
        if doc_hash not in dl_docs:
            dl_docs[doc_hash] = DoclingDocument.load_from_json(doc_store[doc_hash])

        page = dl_docs[doc_hash].pages[page_no]
        img = page.image.pil_image.copy()
        draw = ImageDraw.Draw(img)
        padding = line_width + 2

        for prov in provs:
            bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)
            bbox = bbox.normalized(page.size)
            bbox.l = round(bbox.l * img.width - padding)
            bbox.r = round(bbox.r * img.width + padding)
            bbox.t = round(bbox.t * img.height - padding)
            bbox.b = round(bbox.b * img.height + padding)
            draw.rectangle(xy=bbox.as_tuple(), outline=color, width=line_width)

        annotated.append(img)

    return annotated


# ── Query ─────────────────────────────────────────────────────────────────────

def ask(
    question: str,
    chain,
    doc_store: dict[str, Path],
    *,
    output_dir: Path = Path("."),
) -> str:
    """Run RAG query, save annotated page images, and return the answer."""
    resp = chain.invoke({"input": question})
    answer: str = resp["answer"]

    output_dir.mkdir(parents=True, exist_ok=True)
    images = highlight_sources(resp["context"], doc_store)
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
