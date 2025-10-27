#!/usr/bin/env python3
"""FastAPI-powered web chat interface for the LlamaIndex document assistant."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from chat_cli import (
    configure_models,
    load_existing_index,
)
from llama_index.core import Settings
from llama_index.core.chat_engine import ContextChatEngine


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that grounds answers in the provided documents."
)


@dataclass
class ServerConfig:
    """Runtime configuration for the web server."""

    storage_dir: Path = Path("./storage")
    chroma_dir: Path = Path("./storage/chroma_db")
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    top_k: int = 4
    doc_filter: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8000
    source_count: int = 3


@dataclass
class ChatMessage:
    """Represents a single utterance in the chat history."""

    role: str
    content: str


@dataclass
class SessionState:
    """State tracked per user session."""

    engine: ContextChatEngine
    history: List[ChatMessage] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the assistant.")
    session_id: Optional[str] = Field(
        None,
        description="Existing session identifier; omit to start a new session.",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include supporting sources in the response.",
    )


class SourceInfo(BaseModel):
    source: str
    snippet: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    messages: List[Dict[str, str]]
    sources: List[SourceInfo]


class SessionResponse(BaseModel):
    session_id: str


def create_app(config: ServerConfig) -> FastAPI:
    """Instantiate the FastAPI application with the provided configuration."""
    app = FastAPI(title="LlamaIndex Chat")
    app.state.config = config
    app.state.sessions: Dict[str, SessionState] = {}
    app.state.session_lock = asyncio.Lock()
    app.state.retriever = None

    @app.on_event("startup")
    async def startup_event() -> None:
        try:
            configure_models(config.llm_model, config.embed_model)
            index = load_existing_index(config.storage_dir, config.chroma_dir)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the persisted index. "
                "Ensure incremental_index.py has been run successfully."
            ) from exc

        doc_ids = list(index.docstore.docs.keys())
        retriever_kwargs: Dict[str, Any] = {"similarity_top_k": config.top_k}
        if config.doc_filter:
            filtered_ids = [doc_id for doc_id in doc_ids if config.doc_filter in doc_id]
            if not filtered_ids:
                raise RuntimeError(
                    f"No documents found that match filter '{config.doc_filter}'."
                )
            retriever_kwargs["doc_ids"] = filtered_ids

        app.state.retriever = index.as_retriever(**retriever_kwargs)

    def new_session() -> Tuple[str, SessionState]:
        """Create a new chat session with isolated conversation history."""
        if app.state.retriever is None:
            raise RuntimeError("Retriever not initialized; startup may have failed.")

        engine = ContextChatEngine.from_defaults(
            retriever=app.state.retriever,
            system_prompt=config.system_prompt,
            llm=Settings.llm,
            verbose=False,
        )
        session_id = uuid4().hex
        session = SessionState(engine=engine)
        app.state.sessions[session_id] = session
        return session_id, session

    async def get_or_create_session(
        maybe_session_id: Optional[str],
    ) -> Tuple[str, SessionState]:
        """Return a session by id or create a new one if none provided."""
        async with app.state.session_lock:
            if maybe_session_id and maybe_session_id in app.state.sessions:
                return maybe_session_id, app.state.sessions[maybe_session_id]
            return new_session()

    def extract_sources(nodes: Any, limit: int) -> List[SourceInfo]:
        """Convert LlamaIndex source nodes into serializable metadata."""
        result: List[SourceInfo] = []
        if not nodes:
            return result

        for idx, node_with_score in enumerate(nodes):
            if idx >= limit:
                break
            node = getattr(node_with_score, "node", node_with_score)
            metadata = getattr(node, "metadata", {}) or {}
            source = (
                metadata.get("file_path")
                or metadata.get("source")
                or getattr(node, "ref_doc_id", "Unknown source")
            )
            raw_text = getattr(node, "text", "") or getattr(
                node, "get_content", lambda: ""
            )()
            snippet = " ".join(raw_text.strip().split()) or "[empty snippet]"
            score = getattr(node_with_score, "score", None)
            as_float: Optional[float] = None
            if isinstance(score, (int, float)):
                as_float = float(score)
            result.append(SourceInfo(source=source, snippet=snippet[:200], score=as_float))
        return result

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Serve the single-page chat client."""
        return HTML_PAGE

    @app.get("/api/health")
    async def health() -> Dict[str, str]:
        """Health probe endpoint."""
        status = "ready" if app.state.retriever is not None else "loading"
        return {"status": status}

    @app.post("/api/session", response_model=SessionResponse)
    async def create_session() -> SessionResponse:
        """Create a brand-new chat session."""
        session_id, _ = await get_or_create_session(None)
        return SessionResponse(session_id=session_id)

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> JSONResponse:
        """Handle a user message and return the assistant reply."""
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message must not be empty.")

        try:
            session_id, session = await get_or_create_session(request.session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        async with session.lock:
            session.history.append(ChatMessage(role="user", content=request.message))

            try:
                response = await asyncio.to_thread(session.engine.chat, request.message)
            except Exception as exc:
                session.history.pop()
                raise HTTPException(
                    status_code=500, detail="LLM request failed."
                ) from exc

            answer = getattr(response, "response", None) or str(response)
            session.history.append(ChatMessage(role="assistant", content=answer))

            sources: List[SourceInfo] = []
            if request.include_sources:
                source_nodes = getattr(response, "source_nodes", None) or []
                sources = extract_sources(source_nodes, config.source_count)

        return JSONResponse(
            ChatResponse(
                session_id=session_id,
                response=answer,
                messages=[{"role": msg.role, "content": msg.content} for msg in session.history],
                sources=sources,
            ).model_dump()
        )

    return app


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LlamaIndex Chat</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f4f4;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background: #1f2937;
      color: #fff;
      padding: 12px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    header h1 {
      font-size: 1.1rem;
      margin: 0;
    }
    #new-session {
      background: #2563eb;
      color: #fff;
      border: none;
      padding: 8px 12px;
      border-radius: 4px;
      cursor: pointer;
    }
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      box-sizing: border-box;
    }
    .message {
      margin-bottom: 16px;
      max-width: 720px;
      padding: 12px 16px;
      border-radius: 8px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .message.user {
      background: #2563eb;
      color: #fff;
      margin-left: auto;
    }
    .message.assistant {
      background: #fff;
      color: #111827;
      border: 1px solid #d1d5db;
      box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }
    .sources {
      margin-top: 8px;
      font-size: 0.85rem;
      color: #4b5563;
    }
    .sources strong {
      display: block;
      color: #111827;
      margin-bottom: 4px;
    }
    form {
      background: #fff;
      padding: 16px;
      box-shadow: 0 -1px 4px rgba(0,0,0,0.1);
      display: flex;
      gap: 12px;
    }
    textarea {
      flex: 1;
      resize: none;
      min-height: 60px;
      max-height: 160px;
      padding: 10px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      font: inherit;
    }
    button[type="submit"] {
      background: #2563eb;
      color: #fff;
      border: none;
      padding: 0 20px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
    }
    #status {
      margin: 0;
      font-size: 0.9rem;
    }
    #status.ready {
      color: #10b981;
    }
    #status.loading {
      color: #f59e0b;
    }
    #status.error {
      color: #ef4444;
    }
  </style>
</head>
<body>
  <header>
    <h1>LlamaIndex Chat</h1>
    <div>
      <span id="status" class="loading">Loading...</span>
      <button id="new-session" type="button">New conversation</button>
    </div>
  </header>
  <main id="chat"></main>
  <form id="chat-form">
    <textarea id="message" placeholder="Ask something about your documents..." required></textarea>
    <button type="submit">Send</button>
  </form>
  <script>
    const chatEl = document.getElementById('chat');
    const form = document.getElementById('chat-form');
    const textarea = document.getElementById('message');
    const statusEl = document.getElementById('status');
    const newSessionBtn = document.getElementById('new-session');

    let sessionId = null;
    let isSending = false;

    async function ensureSession() {
      if (sessionId) {
        return sessionId;
      }
      const response = await fetch('/api/session', { method: 'POST' });
      if (!response.ok) {
        throw new Error('Unable to create session.');
      }
      const data = await response.json();
      sessionId = data.session_id;
      return sessionId;
    }

    function appendMessage(role, content, sources = []) {
      const container = document.createElement('div');
      container.className = `message ${role}`;
      container.textContent = content;

      if (role === 'assistant' && sources.length > 0) {
        const srcBlock = document.createElement('div');
        srcBlock.className = 'sources';
        const heading = document.createElement('strong');
        heading.textContent = 'Sources';
        srcBlock.appendChild(heading);
        sources.forEach((src) => {
          const item = document.createElement('div');
          item.textContent = `${src.source}: ${src.snippet}`;
          srcBlock.appendChild(item);
        });
        container.appendChild(srcBlock);
      }

      chatEl.appendChild(container);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    async function sendMessage(event) {
      event.preventDefault();
      if (isSending) {
        return;
      }
      const message = textarea.value.trim();
      if (!message) {
        return;
      }
      textarea.value = '';
      appendMessage('user', message);

      try {
        isSending = true;
        statusEl.textContent = 'Thinking...';
        statusEl.className = 'loading';
        const currentSession = await ensureSession();
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: currentSession,
            message,
            include_sources: true
          })
        });
        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || 'Assistant error.');
        }
        const data = await response.json();
        sessionId = data.session_id;
        appendMessage('assistant', data.response, data.sources || []);
        statusEl.textContent = 'Ready';
        statusEl.className = 'ready';
      } catch (error) {
        appendMessage('assistant', error.message || 'Something went wrong.');
        statusEl.textContent = 'Error';
        statusEl.className = 'error';
      } finally {
        isSending = false;
      }
    }

    async function resetSession() {
      sessionId = null;
      chatEl.innerHTML = '';
      statusEl.textContent = 'Ready';
      statusEl.className = 'ready';
      textarea.focus();
    }

    async function checkHealth() {
      try {
        const response = await fetch('/api/health');
        if (!response.ok) {
          throw new Error();
        }
        const data = await response.json();
        statusEl.textContent = data.status === 'ready' ? 'Ready' : 'Loading...';
        statusEl.className = data.status;
      } catch {
        statusEl.textContent = 'Error';
        statusEl.className = 'error';
      }
    }

    form.addEventListener('submit', sendMessage);
    newSessionBtn.addEventListener('click', resetSession);
    textarea.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        sendMessage(event);
      }
    });

    checkHealth();
    setInterval(checkHealth, 10000);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the LlamaIndex web chat server.")
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("./storage"),
        help="Directory where the index metadata is stored.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("./storage/chroma_db"),
        help="Directory for the persistent Chroma vector store.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI chat model identifier used for responses.",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model identifier used for retrieval.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt passed to the chat engine.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of top-matching nodes retrieved from the index per turn.",
    )
    parser.add_argument(
        "--doc-filter",
        default=None,
        help="Limit retrieval to documents whose ID contains this substring.",
    )
    parser.add_argument(
        "--source-count",
        type=int,
        default=3,
        help="Maximum number of supporting sources to return.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the web server.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn autoreload (development only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ServerConfig(
        storage_dir=args.storage_dir,
        chroma_dir=args.chroma_dir,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        system_prompt=args.system_prompt,
        top_k=args.top_k,
        doc_filter=args.doc_filter,
        host=args.host,
        port=args.port,
        source_count=args.source_count,
    )

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=args.reload,
    )


# Expose an application instance with default configuration for `uvicorn chat_web:app`.
app = create_app(ServerConfig())


if __name__ == "__main__":
    main()
