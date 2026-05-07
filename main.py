from __future__ import annotations

import argparse
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class PipelineConfig:
    pdf_path: Path
    collection_name: str
    qdrant_path: str
    context_window: int
    chunk_size: int
    chunk_overlap: int
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    batch_size: int


@dataclass(frozen=True)
class DoclingElement:
    docling_id: str
    label: str
    page: int | None
    text: str
    item: Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a PDF with Docling, enrich non-text elements with OpenRouter LLM "
            "descriptions, chunk the result, embed it, and store it in Qdrant."
        )
    )
    parser.add_argument("pdf", type=Path, help="PDF file to index.")
    parser.add_argument("--collection", default="pdf_chunks", help="Qdrant collection name.")
    parser.add_argument("--qdrant-path", default="./qdrant_storage", help="Local Qdrant storage path.")
    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Number of neighboring Docling text elements before and after images, equations, and code.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200, help="Maximum text characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=180, help="Overlapping characters between chunks.")
    parser.add_argument(
        "--llm-model",
        default=os.getenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini"),
        help="OpenRouter chat model used for summaries and element explanations.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        help="OpenRouter embedding model.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
        help="Dense vector size for the Qdrant collection.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Documents to add to Qdrant per batch.")
    return parser


def openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-Title"] = title
    return headers


def make_llm(config: PipelineConfig) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for LLM descriptions.")

    return ChatOpenAI(
        model=config.llm_model,
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        temperature=0,
        default_headers=openrouter_headers() or None,
    )


def make_embeddings(config: PipelineConfig) -> OpenAIEmbeddings:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for embeddings.")

    return OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        dimensions=config.embedding_dimensions,
        default_headers=openrouter_headers() or None,
    )


def page_no(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    return getattr(prov[0], "page_no", None)


def response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def item_text(item: Any, doc: Any) -> str:
    if text := getattr(item, "text", None):
        return str(text).strip()
    if hasattr(item, "caption_text"):
        caption = item.caption_text(doc) if callable(item.caption_text) else item.caption_text
        if caption:
            return str(caption).strip()
    for exporter in ("export_to_markdown", "export_to_text"):
        method = getattr(item, exporter, None)
        if callable(method):
            try:
                return str(method(doc)).strip()
            except TypeError:
                return str(method()).strip()
    return ""


def load_docling_elements(pdf_path: Path) -> tuple[Any, list[DoclingElement]]:
    result = DocumentConverter().convert(pdf_path)
    doc = result.document
    elements: list[DoclingElement] = []

    for item, _level in doc.iterate_items():
        label = getattr(getattr(item, "label", None), "value", str(getattr(item, "label", "")))
        elements.append(
            DoclingElement(
                docling_id=getattr(item, "self_ref", ""),
                label=label,
                page=page_no(item),
                text=item_text(item, doc),
                item=item,
            )
        )

    return doc, elements


def summarize_document(llm: ChatOpenAI, doc: Any) -> str:
    text = doc.export_to_text()
    prompt = (
        "Summarize this PDF for downstream retrieval indexing. Focus on topic, methods, "
        "key entities, and conclusions. Keep it under 220 words.\n\n"
        f"{text[:18000]}"
    )
    return response_text(llm.invoke(prompt))


def surrounding_text(elements: list[DoclingElement], index: int, window: int) -> str:
    if window <= 0:
        return ""
    start = max(0, index - window)
    end = min(len(elements), index + window + 1)
    snippets: list[str] = []
    for pos in range(start, end):
        if pos == index:
            continue
        element = elements[pos]
        if element.text:
            snippets.append(f"[{element.label} page {element.page or 'unknown'}] {element.text}")
    return "\n".join(snippets)


def describe_special_element(
    llm: ChatOpenAI,
    element: DoclingElement,
    document_summary: str,
    context: str,
) -> str:
    kind = {
        DocItemLabel.PICTURE.value: "image",
        DocItemLabel.FORMULA.value: "equation",
        DocItemLabel.CODE.value: "code block",
    }.get(element.label, element.label)

    prompt = f"""
    You are preparing a PDF element for vector retrieval.
    Element type: {kind}
    Docling ID: {element.docling_id}
    Page: {element.page or "unknown"}

    Document summary:
    {document_summary}

    Element text, caption, formula, or code if available:
    {element.text or "[No direct text extracted for this element.]"}

    Surrounding document text:
    {context or "[No surrounding text available.]"}

    Write a concise retrieval-friendly explanation of this {kind}. Include what it likely
    represents, important terms from the context, and any formula/code meaning when present.
    Do not invent numeric values or claims that are not supported by the provided text.
    """
    return response_text(llm.invoke(textwrap.dedent(prompt).strip()))


def normalize_element_text(
    llm: ChatOpenAI,
    elements: list[DoclingElement],
    index: int,
    document_summary: str,
    context_window: int,
) -> str:
    element = elements[index]
    special_labels = {
        DocItemLabel.PICTURE.value,
        DocItemLabel.FORMULA.value,
        DocItemLabel.CODE.value,
    }
    if element.label in special_labels:
        return describe_special_element(
            llm=llm,
            element=element,
            document_summary=document_summary,
            context=surrounding_text(elements, index, context_window),
        )
    return element.text


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            split_at = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(" ", start, end))
            if split_at > start + chunk_size // 2:
                end = split_at + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return [chunk for chunk in chunks if chunk]


def build_documents(config: PipelineConfig, llm: ChatOpenAI) -> list[Document]:
    doc, elements = load_docling_elements(config.pdf_path)
    document_summary = summarize_document(llm, doc)
    pdf_name = config.pdf_path.name
    documents: list[Document] = []

    for index, element in enumerate(elements):
        normalized = normalize_element_text(
            llm=llm,
            elements=elements,
            index=index,
            document_summary=document_summary,
            context_window=config.context_window,
        )
        for chunk_index, chunk in enumerate(chunk_text(normalized, config.chunk_size, config.chunk_overlap)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "pdf_name": pdf_name,
                        "docling_id": element.docling_id,
                        "page": element.page,
                        "element_label": element.label,
                        "chunk_index": chunk_index,
                    },
                )
            )

    return documents


def ensure_collection(client: QdrantClient, config: PipelineConfig) -> None:
    existing = {collection.name for collection in client.get_collections().collections}
    if config.collection_name in existing:
        return
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(size=config.embedding_dimensions, distance=Distance.COSINE),
    )


def store_documents(config: PipelineConfig, documents: list[Document]) -> None:
    embeddings = make_embeddings(config)
    client = QdrantClient(path=config.qdrant_path)
    ensure_collection(client, config)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
        embedding=embeddings,
    )

    ids = [
        str(uuid5(NAMESPACE_URL, f"{doc.metadata['pdf_name']}:{doc.metadata['docling_id']}:{doc.metadata['chunk_index']}"))
        for doc in documents
    ]
    for start in range(0, len(documents), config.batch_size):
        end = start + config.batch_size
        vector_store.add_documents(documents=documents[start:end], ids=ids[start:end])


def parse_config() -> PipelineConfig:
    args = build_arg_parser().parse_args()
    return PipelineConfig(
        pdf_path=args.pdf,
        collection_name=args.collection,
        qdrant_path=args.qdrant_path,
        context_window=args.context_window,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        batch_size=args.batch_size,
    )


def main() -> None:
    load_dotenv()
    config = parse_config()
    if not config.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {config.pdf_path}")

    llm = make_llm(config)
    documents = build_documents(config, llm)
    if not documents:
        raise RuntimeError(f"No chunks were produced from {config.pdf_path}")
    store_documents(config, documents)
    print(
        f"Stored {len(documents)} chunks from {config.pdf_path.name} "
        f"in Qdrant collection '{config.collection_name}' at {config.qdrant_path}."
    )


if __name__ == "__main__":
    main()
