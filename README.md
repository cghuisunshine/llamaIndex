# LlamaIndex Project

This project demonstrates how to use LlamaIndex to create, store, and query a vector index for a collection of documents.

## Scripts

*   `query.py`: This is the main script. It reads documents from the `./data` directory, builds a `VectorStoreIndex` using OpenAI's LLM and embedding models, saves the index to the `./storage` directory, and then queries the index.
*   `query_storage.py`: This script loads a pre-existing index from the `./storage` directory and queries it. It's similar to `query.py` but skips the indexing part.
*   `ls_docs.py`: This script loads an index from `./storage` and lists the documents (nodes) that are stored in it.
*   `show_embedding.py`: This script loads an index from `./storage` and displays the embedding vector for a single node.

## How to Use

1.  **Set up your environment:**
    *   Install the required Python packages: `llama-index`, `openai`.
    *   Set your OpenAI API key as an environment variable.

2.  **Add your documents:**
    *   Place the documents you want to index in the `data` directory.

3.  **Build the index:**
    *   Run `python query.py` to build the index and save it to the `storage` directory.

4.  **Query the index:**
    *   Run `python query_storage.py` to query the index with a sample query. You can modify the query in the script.

## Directory Structure

*   `data/`: Contains the documents to be indexed.
*   `storage/`: Stores the vector index.
*   `*.py`: Python scripts for interacting with the LlamaIndex.
