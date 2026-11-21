"""Búsqueda en documentos (Chroma en disco, junto al workflow): la de un RAG Agent y la tool
de un bloque Knowledge, que busca en varios documentos.

La primera vez que se usa, el documento (`documents/<nombre>.txt`, exportado con el
workflow) se parte en fragmentos, se generan sus embeddings y se guardan en
`<carpeta del workflow>/.index/`. Las siguientes ejecuciones solo generan el embedding de la
pregunta. Si cambia el documento o la configuración, se indexa de nuevo. La carpeta `.index/`
es un caché: se puede borrar sin perder nada.
"""

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

from runtime.config import KnowledgeTool, RagNode, provider_key

BATCH = 64
_client = None
_locks: dict[str, asyncio.Lock] = {}


@dataclass
class Chunk:
    index: int
    text: str
    score: float


@dataclass(frozen=True)
class IndexSpec:
    """Cómo se indexa un documento: con qué embeddings y en qué fragmentos."""

    prefix: str
    document_file: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


def rag_spec(node: RagNode) -> IndexSpec:
    return IndexSpec(
        prefix=node.key,
        document_file=node.document_file,
        embedding_provider=node.embedding_provider,
        embedding_model=node.embedding_model,
        chunk_size=node.chunk_size,
        chunk_overlap=node.chunk_overlap,
    )


def _chroma(folder: Path):
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings

        path = folder / ".index"
        path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    return _client


def _embeddings(node: IndexSpec):
    api_key = provider_key(node.embedding_provider)
    if node.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=node.embedding_model, api_key=api_key)
    if node.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = node.embedding_model if node.embedding_model.startswith("models/") else f"models/{node.embedding_model}"
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    raise ValueError(f"El proveedor {node.embedding_provider} no tiene modelos de embeddings.")


def _collection_name(node: IndexSpec, text: str) -> str:
    fingerprint = "|".join(
        [
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            node.embedding_provider,
            node.embedding_model,
            str(node.chunk_size),
            str(node.chunk_overlap),
        ]
    )
    prefix = "".join(c if c.isalnum() else "_" for c in node.prefix)[:40]
    return f"{prefix}_{hashlib.sha256(fingerprint.encode()).hexdigest()[:16]}"


async def _collection(folder: Path, node: IndexSpec, text: str):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    name = _collection_name(node, text)
    async with _locks.setdefault(name, asyncio.Lock()):
        client = _chroma(folder)
        try:
            existing = client.get_collection(name)
        except Exception:
            existing = None
        if existing is not None and (existing.metadata or {}).get("complete"):
            return existing
        if existing is not None:
            client.delete_collection(name)

        splitter = RecursiveCharacterTextSplitter(chunk_size=node.chunk_size, chunk_overlap=node.chunk_overlap)
        chunks = [chunk for chunk in splitter.split_text(text) if chunk.strip()]
        embedder = _embeddings(node)
        collection = client.create_collection(name, configuration={"hnsw": {"space": "cosine"}}, metadata={"file": node.document_file})
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start : start + BATCH]
            vectors = await embedder.aembed_documents(batch)
            collection.add(
                ids=[str(start + i) for i in range(len(batch))],
                documents=batch,
                embeddings=vectors,
                metadatas=[{"index": start + i} for i in range(len(batch))],
            )
        collection.modify(metadata={"file": node.document_file, "complete": True})
        return collection


async def search_index(folder: Path, node: IndexSpec, query: str, top_k: int) -> list[Chunk]:
    text = (folder / node.document_file).read_text(encoding="utf-8")
    collection = await _collection(folder, node, text)
    vector = await _embeddings(node).aembed_query(query)
    result = collection.query(query_embeddings=[vector], n_results=top_k, include=["documents", "metadatas", "distances"])
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        Chunk(index=int((meta or {}).get("index", i)), text=doc, score=round(1 - float(dist), 4))
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances))
    ]


async def search(folder: Path, node: RagNode, query: str) -> list[Chunk]:
    """Los fragmentos del documento de un RAG Agent más parecidos a la pregunta."""
    return await search_index(folder, rag_spec(node), query, node.top_k)


async def search_knowledge(folder: Path, knowledge: KnowledgeTool, query: str) -> str:
    """La tool de un bloque Knowledge: los `top_k` fragmentos más parecidos entre sus documentos."""
    found: list[tuple[Chunk, str]] = []
    for document in knowledge.documents:
        spec = IndexSpec(
            prefix=f"kb_{Path(document.file).stem}",
            document_file=document.file,
            embedding_provider=knowledge.embedding_provider,
            embedding_model=knowledge.embedding_model,
            chunk_size=knowledge.chunk_size,
            chunk_overlap=knowledge.chunk_overlap,
        )
        found += [(chunk, document.name) for chunk in await search_index(folder, spec, query, knowledge.top_k)]
    found = sorted(found, key=lambda item: item[0].score, reverse=True)[: knowledge.top_k]
    if not found:
        return "No se encontraron fragmentos relacionados con la consulta."
    return "\n\n".join(
        f'[{i}] Documento "{name}" · fragmento {chunk.index + 1} · similitud {chunk.score:.2f}\n{chunk.text}'
        for i, (chunk, name) in enumerate(found, start=1)
    )


def context_prompt(node: RagNode, chunks: list[Chunk]) -> str:
    listing = "\n\n".join(
        f"[{i}] (fragmento {chunk.index + 1}, similitud {chunk.score:.2f})\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return (
        f"{node.system_prompt}\n\nFragmentos del documento \"{node.document_name}\" más relacionados con la pregunta:\n\n"
        f"{listing or '(ninguno)'}"
    )
