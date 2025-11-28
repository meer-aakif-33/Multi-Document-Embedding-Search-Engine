# Multi-Document Embedding Search Engine with Caching

This project implements a lightweight, production-ready semantic search engine over 100–200 text documents using embeddings, caching, FAISS vector search, FastAPI, and an optional Streamlit UI.
Below is your text **with all emojis removed and no other changes made**:

---

# Multi-Document Embedding Search Engine with Caching

This project implements a lightweight, production-ready semantic search engine over 100–200 text documents using embeddings, caching, FAISS vector search, FastAPI, and an optional Streamlit UI.

---

## Features

* Preprocessing of raw text documents (cleaning, hashing, metadata)
* Efficient embedding generation using **sentence-transformers/all-MiniLM-L6-v2**
* **SQLite-based embedding cache** — no recomputation if unchanged
* **FAISS vector index**, persisted to `faiss.index`
* Automatic **NumPy cosine similarity fallback** if FAISS unavailable
* **FastAPI `/search` endpoint** for semantic retrieval
* **Ranking explanation** (keyword overlap, ratio, length normalization)
* **Batch embedding with multiprocessing** for fast indexing
* **Streamlit UI** for interactive search
* **Evaluation script** for quality testing
* **Unit tests included**
* Modular, scalable codebase

---

## Folder Structure

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
│   │   ├── embedder.py
│   │   └── batch_embedder.py
│   ├── indexer/
│   │   └── faiss_index.py
│   ├── retriever/
│   │   └── search_engine.py
│   ├── utils/
│   │   ├── cleaning.py
│   │   └── hashing.py
│   └── config.py
│
├── evaluation/
│   └── evaluate.py
│
├── streamlit_app.py
│
├── data/
│   └── docs/        # 100+ .txt documents (ignored by git)
│
├── tests/
│   ├── test_loader.py
│   ├── test_embedder.py
│   ├── test_cache.py
│   └── test_search.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Preprocessing Pipeline

Each document is cleaned and normalized:

✔ Lowercase
✔ Remove HTML tags
✔ Collapse multiple spaces
✔ Compute SHA-256 hash
✔ Extract metadata (filename, length, doc_id)

Implemented in:

```
src/document_loader/loader.py
```

---

## Embedding Generation

Using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Benefits:

* Lightweight (22MB)
* Fast
* High semantic quality

Implemented in:

```
src/embedder/embedder.py
```

---

## Caching System (SQLite)

To avoid recomputing embeddings:

Each cache entry stores:

```
doc_id
sha256_hash_of_cleaned_text
embedding (pickled)
updated_at timestamp
```

Behavior:

* If hash matches → reuse embedding
* If hash differs → regenerate + update cache

Implemented in:

```
src/cache/cache_manager.py
```

---

## Batch Embedding With Multiprocessing

For uncached documents, embeddings are generated using:

```
src/embedder/batch_embedder.py
```

Features:

* Uses multiprocessing.Pool
* Loads model once per worker
* Produces fast embeddings for 100–200 docs
* Integrated in SearchEngine (index_documents)

---

## Vector Search (FAISS + fallback)

Primary engine:

**FAISS IndexFlatIP**
*Based on cosine similarity (with normalized embeddings)*

If FAISS unavailable → fallback to NumPy cosine similarity.

Index persistence:

✔ On startup → load `faiss.index` if exists
✔ After indexing → save updated FAISS index

Implemented in:

```
src/indexer/faiss_index.py
```

---

## Retrieval API (FastAPI)

Endpoint:

```
POST /search
```

Request:

```json
{
  "query": "machine learning basics",
  "top_k": 5
}
```

Response contains:

* doc_id
* preview
* score
* metadata
* ranking explanation

API entrypoint:

```
src/api/main.py
```

---

## Ranking Explanation

Each result includes:

### ✔ Keyword overlap

### ✔ Overlap ratio

### ✔ Length normalization

### ✔ Combined score

Formula:

```
final_score = 0.8 * vector_score + 0.2 * length_norm
```

Implemented in:

```
src/retriever/search_engine.py
```

---

## Running the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the API server

```
uvicorn src.api.main:app --reload
```

### 3. Browse Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## Streamlit UI

Launch the frontend:

```
streamlit run streamlit_app.py
```

Provides:

* Search bar
* Top-K slider
* Score + explanation per result
* Clean, user-friendly layout

---

## Evaluation Script

Run predefined evaluation queries:

```
python evaluation/evaluate.py
```

Validates:

* Ranking quality
* Consistent vector search
* Correct semantic matches

---

## Unit Tests

Located in:

```
tests/
```

**Run unit tests**:

```
pip install pytest
pytest -q
```

## How Caching Works (Detailed)

1. Load documents
2. Compute hash for each cleaned text
3. For each document:

   * If cache has matching hash → load embedding
   * Else → compute embedding and store in cache
4. Build FAISS index from all embeddings
5. Save FAISS index to disk

---

## Design Choices

* **MiniLM** for optimal speed vs accuracy
* **SQLite** for simple, reliable caching
* **FAISS** for high-performance vector search
* **Fallback cosine similarity** ensures cross-platform reliability
* **Modular code** for extensibility and clarity

---

### Already Implemented:

* Streamlit UI
* Persistent FAISS index
* Multiprocessing batch embedding
* Evaluation queries
* Unit tests

### Pending (Optional Bonus):

* Query expansion (WordNet or embedding-based)

---

## Assignment Compliance

### All Mandatory Requirements — **DONE**

### Most Bonus Requirements — **DONE**

Optional: Query Expansion (not included)

---

If you want, I can also:
✔ remove markdown formatting
✔ convert to PDF / DOCX
✔ convert to plain text
✔ extract summary or highlights

---

## 🚀 Features

* Preprocessing of raw text documents (cleaning, hashing, metadata)
* Efficient embedding generation using **sentence-transformers/all-MiniLM-L6-v2**
* **SQLite-based embedding cache** — no recomputation if unchanged
* **FAISS vector index**, persisted to `faiss.index`
* Automatic **NumPy cosine similarity fallback** if FAISS unavailable
* **FastAPI `/search` endpoint** for semantic retrieval
* **Ranking explanation** (keyword overlap, ratio, length normalization)
* **Batch embedding with multiprocessing** for fast indexing
* **Streamlit UI** for interactive search
* **Evaluation script** for quality testing
* **Unit tests included**
* Modular, scalable codebase

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
│   │   ├── embedder.py
│   │   └── batch_embedder.py
│   ├── indexer/
│   │   └── faiss_index.py
│   ├── retriever/
│   │   └── search_engine.py
│   ├── utils/
│   │   ├── cleaning.py
│   │   └── hashing.py
│   └── config.py
│
├── evaluation/
│   └── evaluate.py
│
├── streamlit_app.py
│
├── data/
│   └── docs/        # 100+ .txt documents (ignored by git)
│
├── tests/
│   ├── test_loader.py
│   ├── test_embedder.py
│   ├── test_cache.py
│   └── test_search.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧹 Preprocessing Pipeline

Each document is cleaned and normalized:

✔ Lowercase  
✔ Remove HTML tags  
✔ Collapse multiple spaces  
✔ Compute SHA-256 hash  
✔ Extract metadata (filename, length, doc_id)  

Implemented in:

```
src/document_loader/loader.py
```

---

## ⚡ Embedding Generation

Using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Benefits:

* Lightweight (22MB)
* Fast
* High semantic quality

Implemented in:

```
src/embedder/embedder.py
```

---

## 💾 Caching System (SQLite)

To avoid recomputing embeddings:

Each cache entry stores:

```
doc_id
sha256_hash_of_cleaned_text
embedding (pickled)
updated_at timestamp
```

Behavior:

* If hash matches → reuse embedding  
* If hash differs → regenerate + update cache  

Implemented in:

```
src/cache/cache_manager.py
```

---

## 🔥 Batch Embedding With Multiprocessing

For uncached documents, embeddings are generated using:

```
src/embedder/batch_embedder.py
```

Features:

* Uses multiprocessing.Pool  
* Loads model once per worker  
* Produces fast embeddings for 100–200 docs  
* Integrated in SearchEngine (index_documents)

---

## 🔍 Vector Search (FAISS + fallback)

Primary engine:

**FAISS IndexFlatIP**  
*Based on cosine similarity (with normalized embeddings)*

If FAISS unavailable → fallback to NumPy cosine similarity.

Index persistence:

✔ On startup → load `faiss.index` if exists  
✔ After indexing → save updated FAISS index  

Implemented in:

```
src/indexer/faiss_index.py
```

---

## 🔎 Retrieval API (FastAPI)

Endpoint:

```
POST /search
```

Request:

```json
{
  "query": "machine learning basics",
  "top_k": 5
}
```

Response contains:

* doc_id  
* preview  
* score  
* metadata  
* ranking explanation  

API entrypoint:

```
src/api/main.py
```

---

## 🧠 Ranking Explanation

Each result includes:

### ✔ Keyword overlap  
### ✔ Overlap ratio  
### ✔ Length normalization  
### ✔ Combined score  

Formula:
```
final_score = 0.8 * vector_score + 0.2 * length_norm
```

Implemented in:

```
src/retriever/search_engine.py
```

---

## ▶️ Running the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the API server

```
uvicorn src.api.main:app --reload
```

### 3. Browse Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 🌐 Streamlit UI

Launch the frontend:

```
streamlit run streamlit_app.py
```

Provides:

* Search bar  
* Top-K slider  
* Score + explanation per result  
* Clean, user‑friendly layout  

---

## 📊 Evaluation Script

Run predefined evaluation queries:

```
python evaluation/evaluate.py
```

Validates:

* Ranking quality  
* Consistent vector search  
* Correct semantic matches  

---

## 🧪 Unit Tests

Located in:

```
tests/
```

**Run unit tests**:
```
pip install pytest
pytest -q
```

## 💽 How Caching Works (Detailed)

1. Load documents  
2. Compute hash for each cleaned text  
3. For each document:
   - If cache has matching hash → load embedding  
   - Else → compute embedding and store in cache  
4. Build FAISS index from all embeddings  
5. Save FAISS index to disk  

---

## 🔧 Design Choices

* **MiniLM** for optimal speed vs accuracy  
* **SQLite** for simple, reliable caching  
* **FAISS** for high-performance vector search  
* **Fallback cosine similarity** ensures cross-platform reliability  
* **Modular code** for extensibility and clarity  

---

### ✔ Implemented:
* Streamlit UI  
* Persistent FAISS index  
* Multiprocessing batch embedding  
* Evaluation queries  
* Unit tests  

### Future Scope:
* Query expansion (WordNet or embedding-based)  

---
