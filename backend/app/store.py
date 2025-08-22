import os
import chromadb
from chromadb.config import Settings

_client = None
_collection = None

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ops_copilot")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")

def _init_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    return _client

def get_collection():
    global _collection
    if _collection is None:
        client = _init_client()
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _collection
