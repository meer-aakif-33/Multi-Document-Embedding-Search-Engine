import os

# Path to document folder
DATA_FOLDER = os.environ.get("DATA_FOLDER", "data/docs")

# Path to SQLite cache database
CACHE_DB = os.environ.get("CACHE_DB", "embeddings_cache.db")

# Embedding model to use
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
