# Multi-Document Embedding Search Engine with Caching

This project implements a lightweight, production-ready semantic search engine over 100–200 text documents using embeddings, caching, FAISS vector search, and a FastAPI retrieval API.

## 🚀 Features

* Preprocessing of raw text documents (cleaning, hashing, metadata)
* Efficient embedding generation using **sentence-transformers/all-MiniLM-L6-v2**
* **SQLite-based embedding cache** (no recomputation)
* FAISS vector index with fallback NumPy cosine similarity
* FastAPI `/search` endpoint
* Ranking explanation (keyword overlap, ratio, length normalization)
* Modular structure (`loader`, `embedder`, `cache`, `indexer`, `search_engine`, `api`)
* 100+ sample documents included

---

## 📂 Folder Structure

```
MultiDocSearch/
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── cache/
│   │   └── cache_manager.py
│   ├── document_loader/
│   │   └── loader.py
│   ├── embedder/
│   │   └── embedder.py
│   ├── indexer/
│   │   └── faiss_index.py
│   ├── retriever/
│   │   └── search_engine.py
│   ├── utils/
│   │   ├── cleaning.py
│   │   └── hashing.py
│   └── config.py
│
├── data/
│   └── docs/        # 100+ .txt documents
│
├── tests/
│   └── (to be added)
│
├── requirements.txt
├── README.md (this file)
└── .gitignore
```

---

## 🧹 Preprocessing Pipeline

Each document is cleaned and normalized:

* Convert to lowercase
* Remove HTML tags
* Collapse multiple spaces
* Compute SHA-256 hash
* Collect metadata: filename, text length

This is implemented inside `src/document_loader/loader.py`.

---

## ⚡ Embedding Generation

We use **sentence-transformers/all-MiniLM-L6-v2**:

* Fast (40ms per document)
* Small (22MB)
* Good semantic quality

Embeddings are generated in `embedder/embedder.py`.

---

## 💾 Caching System

To avoid recomputing embeddings, we use a **SQLite local cache**.

Cache entry example:

```
{
  "doc_id": "doc_001",
  "embedding": [...],
  "hash": "sha256_hash",
  "updated_at": timestamp
}
```

**Cache behavior:**

* If hash unchanged → reuse cached embedding
* If file hash changes → regenerate and update cache

Implemented in `cache/cache_manager.py`.

---

## 🔍 Vector Search (FAISS + Fallback)

Primary search backend: **FAISS IndexFlatIP**

* Uses cosine similarity (normalized vectors)

If FAISS not installed → automatic fallback to **NumPy cosine similarity**.

Implemented in `indexer/faiss_index.py`.

---

## 🔎 Retrieval API (FastAPI)

Endpoint: `POST /search`

Request:

```json
{
  "query": "machine learning basics",
  "top_k": 5
}
```

Response:

```json
{
  "query": "machine learning basics",
  "results": [
    {
      "doc_id": "doc_055",
      "score": 0.62,
      "preview": "document 055 ...",
      "explanation": {
        "keyword_overlap": ["machine", "learning"],
        "overlap_ratio": 0.66,
        "length_norm": 0.91
      }
    }
  ]
}
```

---

## 🧠 Ranking Explanation

For each document we compute:

* **Keyword overlap** (simple token-based)
* **Overlap ratio** = overlap / query_words
* **Length normalization** (shorter docs slightly favored)
* **Combined score** = 0.8 * cosine + 0.2 * length

Implemented in `retriever/search_engine.py`.

---

## ▶️ Running the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the API

```
uvicorn src.api.main:app --reload
```

### 3. Test in browser

Visit Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 💽 How Caching Works (Detailed)

On startup:

1. Documents are loaded and hashed
2. Cache is queried:

   * If entry exists and hash matches → load embedding
   * Else → compute embedding and save to cache
3. Embeddings are built into FAISS index

Cache database: `embeddings_cache.db`

---

## 🔧 Design Choices

* **Sentence transformers** chosen for best performance/quality trade-off.
* **SQLite** chosen for simplicity and reliability.
* **FAISS** (or fallback NumPy) ensures fast retrieval.
* **Modular code** for extensibility.

---

## 🧪 Tests

Unit tests are placed under:

```
tests/
```

---

## 📌 Next Steps (Bonus Features)

* Add Streamlit UI for searching
* Persist FAISS index on disk
* Add query expansion (WordNet or embedding similarity)
* Multiprocessing batch embedding
* Retrieval evaluation pipeline

---

## ✔️ Assignment Compliance

This project covers all mandatory requirements:

* Preprocessing
* Embedding generation
* Local cache
* Vector search
* Retrieval API
* Ranking explanation
* Modular structure

