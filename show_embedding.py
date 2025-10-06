# show_embedding.py  (robust for 0.11.x–0.12.x)
from llama_index.core import StorageContext, load_index_from_storage

persist_dir = "./storage"

# Load index + storage
storage = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage)

# Pick a node id from the index structure
node_ids = list(index.index_struct.nodes_dict.keys())
node_id = node_ids[0]
print("node_id:", node_id)

# Try to access embeddings from the vector store in memory
vs = getattr(storage, "vector_store", None) or storage.vector_stores.get("default")
if vs is None:
    raise RuntimeError("No default vector store found in storage_context.")

# Different versions expose the data slightly differently:
emb = None

# 1) SimpleVectorStore with .data.embedding_dict
for attr in ("data", "_data"):
    data = getattr(vs, attr, None)
    if data is not None and hasattr(data, "embedding_dict"):
        emb = data.embedding_dict.get(node_id)
        if emb is not None:
            break

# 2) Some adapters keep items in a dict
if emb is None:
    for attr in ("_items", "items"):
        items = getattr(vs, attr, None)
        if isinstance(items, dict) and node_id in items:
            cand = items[node_id]
            for k in ("embedding", "values", "vector"):
                if isinstance(cand, dict) and isinstance(cand.get(k), list):
                    emb = cand[k]
                    break

# 3) Fall back to a .get / .get_embeddings style method if provided
if emb is None:
    for method_name in ("get", "get_embeddings", "fetch", "get_by_ids"):
        fn = getattr(vs, method_name, None)
        if callable(fn):
            try:
                got = fn(ids=[node_id])
                # normalize a few common return shapes
                if isinstance(got, dict):
                    for k in ("embedding", "values", "vector"):
                        if isinstance(got.get(node_id, {}).get(k), list):
                            emb = got[node_id][k]
                            break
                elif isinstance(got, list) and got:
                    item = got[0]
                    for k in ("embedding", "values", "vector"):
                        if isinstance(getattr(item, k, None), list):
                            emb = getattr(item, k)
                            break
            except Exception:
                pass
        if emb is not None:
            break

if emb is None:
    print("Embedding not found via vector store API/introspection.")
else:
    print("Embedding dim:", len(emb))
    print("First 8 values:", emb[:8])

