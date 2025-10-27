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

## Directory Structure

*   `data/`: Contains the documents to be indexed.
*   `storage/`: Stores the index metadata; the ANN embeddings live in `storage/chroma_db/`.
*   `*.py`: Python scripts for interacting with the LlamaIndex.
