# LlamaIndex Project

This project demonstrates how to use LlamaIndex to create, store, and query a vector index for a collection of documents.

## Scripts

*   `query.py`: This is the main script. It reads documents from the `./data` directory, builds a `VectorStoreIndex` using OpenAI's LLM and embedding models, saves the index to the `./storage` directory, and then queries the index.
*   `query_storage.py`: This script loads a pre-existing index from the `./storage` directory and queries it. It's similar to `query.py` but skips the indexing part.
*   `ls_docs.py`: This script loads an index from `./storage` and lists the documents (nodes) that are stored in it.
*   `show_embedding.py`: This script loads an index from `./storage` and displays the embedding vector for a single node.
*   `incremental_index.py`: Rebuilds the index (or inserts new files) using a persistent Chroma vector store.
*   `chat_cli.py`: Starts an interactive CLI that chats over the stored index; it expects the Chroma-backed index created by `incremental_index.py`.
*   `chat_web.py`: Serves a FastAPI-based web chat UI with per-user conversations backed by the stored index.
*   `mcp_llamaindex_server.py`: Minimal FastMCP server that can load an existing Chroma-backed index or build one from `data/` on demand.

## How to Use

1.  **Set up your environment:**
    *   Install the required Python packages: `llama-index`, `openai`.
    *   Set your OpenAI API key as an environment variable.

2.  **Add your documents:**
    *   Place the documents you want to index in the `data` directory.

3.  **Build the index:**
    *   Run `python incremental_index.py --rebuild` the first time (or whenever you change the contents of `data/`). The script persists metadata in `./storage` and embeddings in `./storage/chroma_db`.

4.  **Query the index:**
    *   Run `python query_storage.py` to query the index with a sample query. You can modify the query in the script.
    *   For a conversational interface, run `python chat_cli.py --show-sources`.
    *   To launch the multi-user web chat UI, run:

        ```bash
        pip install fastapi uvicorn
        python chat_web.py --host 0.0.0.0 --port 8000
        ```

        Then open `http://127.0.0.1:8000` (or the host/port you choose) in a browser. Each browser tab keeps its own session; use the "New conversation" button to start a fresh chat.

## MCP Server


`mcp_llamaindex_server.py` is a lighter-weight variant that starts on port 8001 by default, tries to load a Chroma collection automatically, and includes a `build_index` MCP tool to ingest `.txt`, `.md`, and `.html` files from `data/`.

1. Install dependencies (adjust to your environment) and make sure the persisted index exists by running `python incremental_index.py --rebuild` at least once:

    ```bash
    pip install mcp llama-index chromadb openai
    export OPENAI_API_KEY=...
    ```

2. Start the server, customising flags as needed. Helpful options include:

    * `--system-prompt` to override the assistant persona.
    * `--top-k` and `--source-count` to tune retrieval breadth and returned snippets.
    * `--doc-filter` to restrict retrieval to document IDs containing a substring.

    ```bash
    python llama_mcp_server.py \
      --data-dir ./data \
      --storage-dir ./storage \
      --chroma-dir ./storage/chroma_db \
      --llm-model gpt-4o-mini \
      --embed-model text-embedding-3-small \
      --top-k 4 \
      --source-count 3
    ```

3. Register the server with your MCP host so it runs over stdio. For ChatGPT:

    
4. When connected, hosts can list/read the shared files and call `chat_with_llama_index`. Input parameters:

    * `message` (string, required) — user query.
    * `include_sources` (boolean, optional, default `true`) — return supporting snippets.
    * `session_id` (string, optional) — reuse from a prior response to continue the conversation.

    ```json
    {
      "message": "What do the training outlines say about safety requirements?",
      "include_sources": true
    }
    ```

   Responses contain:

    * `answer` — the assistant reply.
    * `sources` — up to `--source-count` grounded snippets.
    * `messages` — running conversation context.
    * `session_id` — reference for the next turn.

   Answers are generated through the same retrieval pipeline used in `chat_web.py`, so they stay grounded in the indexed documents.

## Directory Structure

*   `data/`: Contains the documents to be indexed.
*   `storage/`: Stores the index metadata; the ANN embeddings live in `storage/chroma_db/`.
*   `*.py`: Python scripts for interacting with the LlamaIndex.
