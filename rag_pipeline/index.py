import os
from typing import List, Dict, Any
import chromadb
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class Indexer:
    persist_dir: str = "chroma"

    def __post_init__(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def get_or_create(self, name: str):
        return self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def embed(self, texts: List[str]):
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def add_chunks(self, collection_name: str, chunks: List[Dict[str, Any]], episode_id: str, meta_extra: Dict[str, Any]):
        col = self.get_or_create(collection_name)
        ids = [f"{episode_id}:{i}" for i in range(len(chunks))]
        texts = [c["text"] for c in chunks]
        embeddings = self.embed(texts)
        metadatas = [ {"episode_id": episode_id, "start": c["start"], "end": c["end"], **meta_extra} for c in chunks ]
        col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def query(self, collection_name: str, query_text: str, top_k: int = 8):
        col = self.get_or_create(collection_name)
        qemb = self.embed([query_text])[0]
        res = col.query(query_embeddings=[qemb], n_results=top_k, include=["documents", "metadatas", "distances", "embeddings", "ids"])
        return res
