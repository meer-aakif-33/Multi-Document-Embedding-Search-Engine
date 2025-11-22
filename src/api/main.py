#main.py
from ..indexer.faiss_index import FaissIndex
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn

from ..document_loader.loader import load_documents
from ..cache.cache_manager import CacheManager
from ..embedder.embedder import Embedder
from ..retriever.search_engine import SearchEngine


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


app = FastAPI()
ENGINE: SearchEngine = None


@app.on_event("startup")
def startup():
    global ENGINE
    data_folder = os.environ.get("DATA_FOLDER", "data/docs")
    docs = load_documents(data_folder)
    cache = CacheManager(os.environ.get("CACHE_DB", "embeddings_cache.db"))
    embedder = Embedder()
    dim = embedder.embed("test").shape[0]

    ENGINE = SearchEngine(embedder, cache, dim, index_path="faiss.index")
    ENGINE.index_documents(docs)

@app.post("/search")
def search(req: SearchRequest):
    global ENGINE
    if ENGINE is None:
        raise HTTPException(status_code=500, detail="Search engine is not initialized")

    return ENGINE.search(req.query, req.top_k)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
