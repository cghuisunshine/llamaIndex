# Change Log

## Recent Updates

- Added `chat_cli.py` to provide a command-line interface for chatting with the stored LlamaIndex corpus.
- Introduced `incremental_index.py` to build or incrementally update the index and persist results.
- Migrated from the JSON vector store to a persistent Chroma ANN store and updated all scripts to use it.
- Wrapped Chroma queries with a custom adapter so empty metadata filters no longer crash retrieval.
- Hardened index rebuild logic to recreate missing docstore data and display newly indexed documents.
- Extended the chat CLI with `--top-k` and `--doc-filter` flags for finer retrieval control.
- Regenerated embeddings after debugging to ensure the Huntington’s transcript surfaced correctly.
