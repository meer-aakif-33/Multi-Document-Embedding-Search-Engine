"""
src/embedder/batch_embedder.py

Batch embedding utility using multiprocessing to speed up encoding of many documents.

Usage:
    from src.embedder.batch_embedder import embed_batch_multiprocess
    embs = embed_batch_multiprocess(texts, model_name='all-MiniLM-L6-v2', batch_size=32, n_workers=4)

Notes:
- Uses a process pool with a model loaded once per worker (via initializer).
- On Windows, the main module that calls multiprocessing must be guarded by
  if __name__ == '__main__': when running as a script. When used as an imported
  module from FastAPI startup this is fine because the pool is created inside
  a function.
- This function returns a single numpy array of shape (len(texts), dim).
"""
#batch_embedder
from typing import List, Optional
import math
import numpy as np
from multiprocessing import Pool

# We'll lazily import SentenceTransformer inside worker initializer to avoid
# pickling a large model object.
_model = None


def _init_worker(model_name: str):
    """Initializer for worker processes: load the sentence-transformers model into a global variable."""
    global _model
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required but not installed") from e
    _model = SentenceTransformer(model_name)


def _worker_encode(chunk_texts: List[str]) -> np.ndarray:
    """Encode a chunk of texts using the worker's global model."""
    global _model
    if _model is None:
        raise RuntimeError("Worker model is not initialized")
    # convert_to_numpy ensures we receive numpy arrays (not lists)
    embs = _model.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True)
    return embs


def _chunkify(lst: List, n: int) -> List[List]:
    """Split list lst into n approximately equal chunks."""
    if n <= 0:
        return [lst]
    k, m = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        chunks.append(lst[start:end])
        start = end
    return [c for c in chunks if c]


def embed_batch_multiprocess(texts: List[str], model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32, n_workers: Optional[int] = None) -> np.ndarray:
    """Embed a list of texts using multiple processes.

    Args:
        texts: list of strings to embed
        model_name: sentence-transformers model name
        batch_size: per-worker internal batching (unused here but kept for API compatibility)
        n_workers: number of worker processes. If None, uses os.cpu_count() or 2.

    Returns:
        numpy array of shape (len(texts), dim)
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if n_workers is None:
        import os
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    # If small number of texts, fall back to single-process encode to avoid overhead
    if len(texts) < 2 * n_workers:
        # single-process
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # split into chunks per worker
    chunks = _chunkify(texts, n_workers)

    # Use multiprocessing pool with initializer that loads the model per worker
    with Pool(processes=len(chunks), initializer=_init_worker, initargs=(model_name,)) as pool:
        # map each chunk to a worker
        results = pool.map(_worker_encode, chunks)

    # results is list of numpy arrays; concatenate preserving original order
    embs = np.vstack(results)
    return embs


if __name__ == '__main__':
    # small demo when run as a script
    sample = [f"This is sample text {i}" for i in range(200)]
    embs = embed_batch_multiprocess(sample, n_workers=4)
    print('Embeddings shape:', embs.shape)
