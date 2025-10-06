# ls_docs.py
from collections.abc import Mapping, Iterable
from llama_index.core import StorageContext, load_index_from_storage

persist_dir = "./storage"

# 1) Load index
storage = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage)

# 2) Get node IDs (VectorStoreIndex has index_struct.nodes_dict)
index_struct = getattr(index, "index_struct", None)
if index_struct is None or not hasattr(index_struct, "nodes_dict"):
    raise RuntimeError("This script assumes a VectorStoreIndex with nodes_dict.")

node_ids = list(index_struct.nodes_dict.keys())

# 3) Fetch nodes (return type varies by version)
nodes_result = index.docstore.get_nodes(node_ids)

if isinstance(nodes_result, Mapping):          # old: {node_id: Node}
    nodes = list(nodes_result.values())
elif isinstance(nodes_result, list):           # newer: [Node, ...]
    nodes = nodes_result
elif isinstance(nodes_result, Iterable):       # fallback: any iterable of Nodes
    nodes = list(nodes_result)
else:
    # very old APIs: no bulk method — fall back to per-node get
    nodes = [index.docstore.get_node(nid) for nid in node_ids]

print(f"Total nodes: {len(nodes)}\n")

# 4) Pretty print
for i, n in enumerate(nodes, 1):
    src = (getattr(n, "metadata", {}) or {}).get("file_path") \
          or (getattr(n, "metadata", {}) or {}).get("source") \
          or getattr(n, "ref_doc_id", None) \
          or "UNKNOWN"
    text = (getattr(n, "text", "") or "").replace("\n", " ")[:120]
    print(f"{i:03d}  node_id={getattr(n, 'node_id', 'NA')}  source={src}  text='{text}…'")

