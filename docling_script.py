"""Visual Grounding RAG — Docling + Qdrant + OpenRouter + LangChain."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import mkdtemp

from PIL import Image, ImageDraw
from docling.chunking import DocMeta, HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/Besen_2016.pdf"),
]
COLLECTION = "visual_grounding"
QDRANT_PATH = "./qdrant_storage"
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

# ── Document Converter ────────────────────────────────────────────────────────

def make_converter() -> DocumentConverter:
    """Converter with page image generation enabled for visual grounding."""
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    generate_page_images=True,
                    images_scale=2.0,
                )
            )
        }
    )

# ── Indexing ──────────────────────────────────────────────────────────────────

def index(
    sources: list[Path],
    *,
    collection: str = COLLECTION,
    qdrant_path: str = QDRANT_PATH,
    drop_old: bool = False,
) -> tuple[dict[str, Path], QdrantVectorStore]:
    """
    Convert PDFs, chunk them, embed with OpenRouter, and store in Qdrant.

    Returns:
        doc_store: maps binary_hash → path of saved JSON (for visual grounding)
        vector_store: populated QdrantVectorStore
    """
    converter = make_converter()
    chunker = HybridChunker()

    doc_store: dict[str, Path] = {}
    store_root = Path(mkdtemp())
    lc_docs: list[Document] = []

    for source in sources:
        print(f"  Converting {source.name}…")
        dl_doc = converter.convert(source=str(source)).document

        # Persist full document (including page images) for later visual grounding
        json_path = store_root / f"{dl_doc.origin.binary_hash}.json"
        dl_doc.save_as_json(json_path)
        doc_store[dl_doc.origin.binary_hash] = json_path

        # Chunk and wrap as LangChain Documents; store provenance in metadata
        for chunk in chunker.chunk(dl_doc):
            lc_docs.append(
                Document(
                    page_content=chunk.text,
                    metadata={"dl_meta": chunk.meta.model_dump(mode="json")},
                )
            )

    print(f"  Created {len(lc_docs)} chunks from {len(sources)} document(s).")

    api_key = os.environ["OPENROUTER_API_KEY"]
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        dimensions=EMBED_DIM,
    )

    client = QdrantClient(path=qdrant_path)

    if drop_old:
        try:
            client.delete_collection(collection)
        except Exception:
            pass

    if collection not in {c.name for c in client.get_collections().collections}:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )
    vector_store.add_documents(lc_docs)
    return doc_store, vector_store

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
    line_width: int = 3,
) -> list[Image.Image]:
    """
    Draw bounding boxes on the page images for every retrieved chunk.

    For each retrieved chunk the provenance (page number + bbox) is read from
    the stored DocMeta.  The source PDF JSON is loaded from doc_store, the
    correct page image is taken, and bounding boxes are drawn on it.
    """
    # Group bounding boxes by (doc_hash, page_no)
    page_entries: dict[tuple[str, int], list] = {}
    for lc_doc in context_docs:
        meta = DocMeta.model_validate(lc_doc.metadata["dl_meta"])
        for doc_item in meta.doc_items:
            for prov in doc_item.prov or []:
                key = (meta.origin.binary_hash, prov.page_no)
                page_entries.setdefault(key, []).append((meta.origin.binary_hash, prov))

    annotated: list[Image.Image] = []
    for (doc_hash, page_no), entries in page_entries.items():
        if doc_hash not in doc_store:
            continue

        dl_doc = DoclingDocument.load_from_json(doc_store[doc_hash])
        page = dl_doc.pages[page_no]
        img = page.image.pil_image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for _, prov in entries:
            # Convert to top-left origin, then normalize to [0, 1]
            bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)
            nb = bbox.normalized(page.size)
            draw.rectangle(
                [(nb.l * w, nb.t * h), (nb.r * w, nb.b * h)],
                outline=color,
                width=line_width,
            )

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
    print("Indexing documents…")
    doc_store, vector_store = index(SOURCES, drop_old=True)

    chain = build_chain(vector_store)

    question = "What are the main findings about sepsis treatment?"
    print(f"\nQ: {question}")
    answer = ask(question, chain, doc_store, output_dir=Path("grounding_output"))
    print(f"\nA: {answer}")
