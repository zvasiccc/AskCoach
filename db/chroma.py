import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

class ChromaDBManager:
    _bm25_cache = {}

    def __init__(self):
        self.persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def get_coach_collection(self, coach_id: str):
        collection_name = f"coach_{coach_id}"
        return self.client.get_or_create_collection(name=collection_name)

    def add_to_collection(self, coach_id: str, ids: list, documents: list, metadatas: list, embeddings: list):
        if not ids:
            return
        collection = self.get_coach_collection(coach_id)
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def invalidate_cache(self, coach_id: str):
        for key in [k for k in self._bm25_cache if k[0] == coach_id]:
            del self._bm25_cache[key]

    def delete_collection(self, coach_id: str):
        collection_name = f"coach_{coach_id}"
        try:
            self.client.delete_collection(name=collection_name)
            self.invalidate_cache(coach_id)
            return True
        except Exception:
            return False

    def list_coaches(self):
        collections = self.client.list_collections()
        return [c.name.replace("coach_", "") for c in collections]