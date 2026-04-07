# API Reference

This document provides a concise reference for the key Python classes and functions used in the **RAG with Qdrant and Groq** notebook. The notebook is self‑contained, but understanding the underlying APIs helps you extend or adapt the pipeline.

---

## 1. Qdrant Client (`qdrant_client`)

### Import
```python
from qdrant_client import QdrantClient
```

### `QdrantClient`
Creates a client to interact with a Qdrant instance (cloud or local).

| Parameter | Type | Description |
|-----------|------|-------------|
| `host` | `str` | Qdrant endpoint (e.g., `"your-instance.qdrant.io"`). |
| `port` | `int` | Port number (default `6333`). |
| `api_key` | `str` | Authentication token for Qdrant Cloud. |
| `prefer_grpc` | `bool` | Use gRPC if available (faster). |

### Common Methods
| Method | Signature | Description |
|--------|-----------|-------------|
| `create_collection` | `client.create_collection(collection_name: str, vectors_config: dict)` | Creates a collection to store vectors. |
| `upsert` | `client.upsert(collection_name: str, points: List[PointStruct])` | Inserts or updates vectors (documents). |
| `search` | `client.search(collection_name: str, query_vector: List[float], limit: int = 5)` | Retrieves the `limit` most similar points. |
| `delete_collection` | `client.delete_collection(collection_name: str)` | Removes an entire collection. |

**Typical Usage in the Notebook**
```python
client = QdrantClient(host=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
client.create_collection(
    collection_name="rag_collection",
    vectors_config={"size": 384, "distance": "Cosine"}
)
# Upsert documents
client.upsert(
    collection_name="rag_collection",
    points=[
        PointStruct(id=doc_id, vector=embedding, payload={"text": doc_text})
        for doc_id, (embedding, doc_text) in enumerate(zip(embeddings, docs))
    ]
)
# Search
hits = client.search(
    collection_name="rag_collection",
    query_vector=query_embedding,
    limit=3
)
```
---

## 2. Sentence‑Transformers (`sentence_transformers`)

### Import
```python
from sentence_transformers import SentenceTransformer
```

### `SentenceTransformer`
Loads a pre‑trained transformer model for generating dense embeddings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name_or_path` | `str` | Model identifier, e.g., `"sentence-transformers/all-MiniLM-L6-v2"`. |
| `device` | `str` | Execution device (`"cpu"` or `"cuda"`). |

### Common Methods
| Method | Signature | Description |
|--------|-----------|-------------|
| `encode` | `model.encode(texts: List[str] | str, batch_size: int = 32, normalize_embeddings: bool = False)` | Returns a NumPy array of embeddings. |

**Typical Usage**
```python
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Single document
doc_embedding = embedder.encode(document)
# Batch of documents
doc_embeddings = embedder.encode(documents, batch_size=64, normalize_embeddings=True)
```
---

## 3. Groq Client (`groq`) – Inference API

### Install
```bash
pip install groq
```

### Import & Initialization
```python
from groq import Groq
client = Groq(api_key=GROQ_API_KEY)
```

### `client.chat.completions.create`
Sends a chat‑style request to the Groq endpoint.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model name, e.g., `"groq/openai-gpt-oss-120b"`. |
| `messages` | `list[dict]` | Conversation history. Each dict has `role` (`"system"`, `"user"`, `"assistant"`) and `content`. |
| `temperature` | `float` | Sampling temperature (default `0.0`). |
| `max_tokens` | `int` | Maximum tokens in the response. |

**Typical Usage in the Notebook**
```python
response = client.chat.completions.create(
    model="groq/openai-gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer the question using the following context:\n{retrieved_context}\n\nQuestion: {user_query}"}
    ],
    temperature=0.0,
    max_tokens=512,
)
answer = response.choices[0].message.content
```
---

## 4. Utility Functions (Defined in the Notebook)

| Function | Purpose |
|----------|---------|
| `load_documents(path: str) -> List[str]` | Reads plain‑text files from a directory and returns a list of document strings. |
| `embed_documents(docs: List[str]) -> np.ndarray` | Uses the `SentenceTransformer` instance to generate embeddings for a batch of documents. |
| `upsert_to_qdrant(embeddings: np.ndarray, docs: List[str])` | Wraps each embedding into a `PointStruct` and upserts them into the Qdrant collection. |
| `retrieve_context(query: str, top_k: int = 3) -> str` | Embeds the query, searches Qdrant, and concatenates the top‑k retrieved document texts. |
| `generate_answer(context: str, query: str) -> str` | Sends the combined context and query to Groq and returns the model’s answer. |

These helpers keep the notebook tidy and can be copied into a standalone Python script if you wish to run the pipeline outside of Colab.

---

## 5. End‑to‑End Example (Python Script Skeleton)
```python
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# 1️⃣ Initialise clients
qdrant = QdrantClient(host=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY"))
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2️⃣ Create collection (run once)
qdrant.create_collection(
    collection_name="rag_collection",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

# 3️⃣ Ingest documents (example)
docs = ["Document 1 text...", "Document 2 text..."]
embeddings = embedder.encode(docs, normalize_embeddings=True)
points = [models.PointStruct(id=i, vector=emb.tolist(), payload={"text": doc})
          for i, (emb, doc) in enumerate(zip(embeddings, docs))]
qdrant.upsert(collection_name="rag_collection", points=points)

# 4️⃣ Query loop
while True:
    query = input("Ask a question (or 'exit'): ")
    if query.lower() in {"exit", "quit"}:
        break
    q_emb = embedder.encode(query, normalize_embeddings=True).tolist()
    hits = qdrant.search(collection_name="rag_collection", query_vector=q_emb, limit=3)
    context = "\n\n".join(hit.payload["text"] for hit in hits)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers based on provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    resp = groq.chat.completions.create(model="groq/openai-gpt-oss-120b", messages=messages, temperature=0.0)
    print("Answer:", resp.choices[0].message.content)
```

---

## 6. References
- **Qdrant Documentation** – https://qdrant.tech/documentation/
- **Sentence‑Transformers** – https://www.sbert.net/
- **Groq API Reference** – https://groq.com/docs/

---

*End of API Reference.*
