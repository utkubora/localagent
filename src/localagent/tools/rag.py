"""
Local retrieval-augmented generation.

Chunks workspace documents into a persistent Chroma collection stored on disk
and exposes tools to index, search, and manage them:

    pip install chromadb

Embeddings run locally through Chroma's bundled ONNX MiniLM model (downloaded
once, cached thereafter) — no API key required, and independent of whichever
chat backend (Anthropic or local transformers) the agent is using.
"""

from __future__ import annotations

from pathlib import Path

from .baseTool import tool
from .tools import WORKSPACE, _safe_path

VECTOR_STORE_DIR = Path("./vectorstore").resolve()
_COLLECTION_NAME = "knowledge"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings

        VECTOR_STORE_DIR.mkdir(exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(_COLLECTION_NAME)
    return _collection


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping windows, preferring to break on a paragraph,
    then sentence, then word boundary so ideas aren't split mid-word."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


@tool
def index_document(filename: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> str:
    """Read a text file from the agent's workspace, split it into overlapping
    chunks, and add it to the local vector knowledge base so it can later be
    found with search_knowledge. Re-indexing the same filename replaces its
    previously indexed chunks.

    Args:
        filename: Path relative to the workspace root, e.g. "docs/manual.md".
        chunk_size: Maximum characters per chunk. Defaults to 1000.
        chunk_overlap: Characters shared between consecutive chunks, so an idea
            near a boundary isn't lost. Defaults to 150.
    """
    text = _safe_path(filename).read_text(encoding="utf-8")
    chunks = _chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        return f"{filename} produced no chunks (empty file?)."

    collection = _get_collection()
    collection.delete(where={"source": filename})
    collection.add(
        ids=[f"{filename}::{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))],
    )
    return f"Indexed {filename}: {len(chunks)} chunk(s)."


@tool
def search_knowledge(query: str, n_results: int = 5) -> str:
    """Search the local vector knowledge base for the chunks most relevant to
    a query and return them together with their source file. Use this before
    answering questions about documents indexed with index_document, instead
    of guessing at their contents.

    Args:
        query: The question or topic to search for.
        n_results: Maximum number of chunks to return. Defaults to 5.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return "The knowledge base is empty. Index documents first with index_document."

    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
    docs, metas, distances = results["documents"][0], results["metadatas"][0], results["distances"][0]
    if not docs:
        return "No relevant results found."

    parts = [
        f"[{meta['source']} #{meta['chunk']}] (distance={dist:.3f})\n{doc}"
        for doc, meta, dist in zip(docs, metas, distances)
    ]
    return "\n\n---\n\n".join(parts)


@tool
def list_indexed_documents() -> str:
    """List the distinct source files currently in the knowledge base, with how
    many chunks each one contributed. Use this to check what's already indexed
    before deciding whether to call index_document again."""
    collection = _get_collection()
    if collection.count() == 0:
        return "(empty)"

    counts: dict[str, int] = {}
    for meta in collection.get()["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return "\n".join(f"{src} ({n} chunk{'s' if n != 1 else ''})" for src, n in sorted(counts.items()))


@tool
def remove_document(filename: str) -> str:
    """Remove all indexed chunks belonging to one file from the knowledge base.

    Args:
        filename: The filename previously passed to index_document.
    """
    _get_collection().delete(where={"source": filename})
    return f"Removed {filename} from the knowledge base."
