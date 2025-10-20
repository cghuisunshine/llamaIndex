python - <<'PY'
import chromadb
TARGET_PATH = "/Users/chenguagnghui/Term3/WebProject2/llama_index/data/2025-09-24-17-55-39.txt"

client = chromadb.PersistentClient(path="storage/chroma_db")
collection = client.get_collection("llama_index_docs")

# Fetch a few nodes for this file
result = collection.get(
    where={"doc_id": TARGET_PATH},
    limit=5,
    include=["documents", "metadatas"]
)
print("Matched nodes:", len(result["ids"]))
for doc, meta in zip(result["documents"], result["metadatas"]):
    print(meta.get("doc_id", meta.get("file_path")), "→", doc[:160], "…")

# Optional: semantic search to double-check retrieval
search = collection.query(
    query_texts=["breaking news Huntington's disease has been successfully"],
    n_results=3
)
print("Text search hits:", search["ids"])
PY

