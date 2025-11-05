#!/usr/bin/env python3
# (content identical to previous message; re-emitting to ensure file is written)

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("mcp_llamaindex_min")

PORT = int(os.environ.get("MCP_PORT", "8001"))
APP_NAME = os.environ.get("MCP_APP_NAME", "llamaindex-mcp")
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "./storage"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", "./storage/chroma_db"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "documents")
SIMILARITY_TOP_K = int(os.environ.get("SIMILARITY_TOP_K", "5"))

INDEX = None
QUERY_ENGINE = None

def _load_existing_index(persist_dir: Path, chroma_dir: Path):
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Chroma v0.6 returns list of NAMES (str) from list_collections(). Older versions returned objects.
    try:
        colls = client.list_collections()
    except TypeError:
        colls = client.list_collections()

    selected_name = None
    if isinstance(colls, list) and colls:
        first = colls[0]
        if isinstance(first, str):
            selected_name = first
        else:
            selected_name = getattr(first, "name", None)

    collection = None
    # 1) Prefer explicitly provided env collection name if available
    if COLLECTION_NAME:
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            collection = None

    # 2) Fall back to the first existing collection
    if collection is None and selected_name:
        try:
            collection = client.get_collection(selected_name)
        except Exception:
            collection = None

    # 3) Last resort: create or get the env-provided name
    if collection is None:
        collection = client.get_or_create_collection(COLLECTION_NAME or "documents")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))

    try:
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    except Exception:
        return load_index_from_storage(storage_context)

def _build_index_from_files(data_dir: Path, persist_dir: Path, chroma_dir: Path):
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    docs: List[Document] = []
    exts = {".txt", ".md", ".html", ".pdf", ".json"}
    if data_dir.exists():
        for p in data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                docs.append(Document(text=text, metadata={"path": str(p)}))

    if not docs:
        raise FileNotFoundError(f"No supported documents found in {data_dir} (expected .txt/.md/.html).")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(persist_dir))

    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    try:
        index.storage_context.persist(persist_dir=str(persist_dir))
    except Exception as e:
        log.warning(f"Persist skipped or failed: {e}")
    return index

def _make_query_engine(index, similarity_top_k: int):
    as_qe = getattr(index, "as_query_engine", None)
    if callable(as_qe):
        return as_qe(similarity_top_k=similarity_top_k)
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    return RetrieverQueryEngine(retriever=retriever)

def init_resources():
    global INDEX, QUERY_ENGINE
    try:
        INDEX = _load_existing_index(STORAGE_DIR, CHROMA_DIR)
        QUERY_ENGINE = _make_query_engine(INDEX, SIMILARITY_TOP_K)
        log.info("Index and query engine initialized from existing Chroma store.")
    except Exception as e:
        INDEX = None
        QUERY_ENGINE = None
        log.warning(f"No existing index loaded at startup: {e}. Use build_index tool to create one.")

mcp = FastMCP(APP_NAME)

@mcp.tool(description="Status of the LlamaIndex/Chroma setup")
def index_status() -> Dict[str, Any]:
    return {
        "storage_dir": str(STORAGE_DIR.resolve()),
        "chroma_dir": str(CHROMA_DIR.resolve()),
        "data_dir": str(DATA_DIR.resolve()),
        "collection": COLLECTION_NAME,
        "index_loaded": INDEX is not None,
        "engine_ready": QUERY_ENGINE is not None,
    }

@mcp.tool(description="Build the index from DATA_DIR (.txt/.md/.html), then load it")
def build_index(data_dir: Optional[str] = None) -> Dict[str, Any]:
    global INDEX, QUERY_ENGINE
    src = Path(data_dir) if data_dir else DATA_DIR
    try:
        index = _build_index_from_files(src, STORAGE_DIR, CHROMA_DIR)
        INDEX = index
        QUERY_ENGINE = _make_query_engine(INDEX, SIMILARITY_TOP_K)
        return {"ok": True, "built_from": str(src.resolve()), "docs_indexed": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@mcp.tool(description="List document IDs in the index (best-effort)")
def list_documents() -> List[str]:
    global INDEX
    if not INDEX:
        return []
    try:
        return list(getattr(INDEX, "docstore").docs.keys())  # type: ignore[attr-defined]
    except Exception:
        return []

@mcp.tool(description="Fetch a document by ID (text + metadata)")
def get_document(doc_id: str) -> Dict[str, Any]:
    global INDEX
    if not INDEX:
        return {}
    try:
        node = getattr(INDEX, "docstore").docs.get(doc_id)  # type: ignore[attr-defined]
        if node is None:
            return {}
        return {
            "doc_id": doc_id,
            "text": getattr(node, "get_content", lambda: None)() or getattr(node, "text", None),
            "metadata": getattr(node, "metadata", {}),
        }
    except Exception:
        return {}

@mcp.tool(description="Semantic search for nodes; returns matches with scores")
def search_index(query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    global INDEX
    if not INDEX:
        return {"matches": []}
    k = int(top_k or SIMILARITY_TOP_K)
    try:
        retriever = getattr(INDEX, "as_retriever", None)
        if callable(retriever):
            r = retriever(similarity_top_k=k)
            nodes = r.retrieve(query)
        else:
            from llama_index.core.retrievers import VectorIndexRetriever
            r = VectorIndexRetriever(index=INDEX, similarity_top_k=k)
            nodes = r.retrieve(query)

        matches = []
        for n in nodes:
            node = getattr(n, "node", n)
            matches.append({
                "node_id": getattr(n, "node_id", getattr(n, "id_", None)),
                "score": getattr(n, "score", None),
                "text": getattr(node, "get_content", lambda: None)() or getattr(node, "text", None),
                "metadata": getattr(node, "metadata", {}),
            })
        return {"matches": matches}
    except Exception as e:
        return {"error": str(e), "matches": []}


@mcp.tool(description="Ask a question; returns synthesized answer with sources (best-effort)")
def ask(question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    global QUERY_ENGINE
    if not QUERY_ENGINE:
        return {"answer": "", "sources": [], "error": "QUERY_ENGINE not initialized"}

    try:
        # Try modern call without kwargs first
        if top_k is None:
            resp = QUERY_ENGINE.query(question)
        else:
            try:
                # Try legacy behavior
                resp = QUERY_ENGINE.query(question, similarity_top_k=int(top_k))
            except TypeError:
                # Fallback: call without the kwarg
                resp = QUERY_ENGINE.query(question)

        text = getattr(resp, "response", None) or str(resp)
        source_nodes = getattr(resp, "source_nodes", []) or []
        sources = []
        for sn in source_nodes:
            node = getattr(sn, "node", sn)
            sources.append({
                "doc_id": getattr(node, "doc_id", None),
                "score": getattr(sn, "score", None),
                "text": (getattr(node, "get_content", lambda: None)() or
                         getattr(node, "text", None)),
                "metadata": getattr(node, "metadata", {}),
            })
        return {"answer": text, "sources": sources}
    except Exception as e:
        return {"answer": "", "sources": [], "error": str(e)}


mcp_http = mcp.streamable_http_app()

@asynccontextmanager
async def lifespan(_app):
    async with mcp.session_manager.run():
        init_resources()
        yield

app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/_health", lambda req: PlainTextResponse("ok")),
        Mount("/", app=mcp_http),
    ],
)

def main():
    print(f"🚀 {APP_NAME} MCP at http://127.0.0.1:{PORT}/mcp  (health: /_health)")
    uvicorn.run(app, host="127.0.0.1", port=PORT)

if __name__ == "__main__":
    main()
