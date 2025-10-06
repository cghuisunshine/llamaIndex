from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1) Choose LLM + embeddings
Settings.llm = OpenAI(model="gpt-4o-mini")                # any chat-capable model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 2) LATER: LOAD (no re-embedding needed)
storage_ctx = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_ctx)

# 3) Get a query engine and ask something
qe = index.as_query_engine(response_mode="compact")       # "compact" = concise answers
resp = qe.query("Give me a 3-bullet summary of the key ideas across these docs.")
print(resp)

