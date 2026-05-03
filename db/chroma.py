import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv

load_dotenv()

class ChromaDBManager:
    def __init__(self):
        #fizicka lokacija baze na disku
        self.persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def get_coach_collection(self, coach_id: str):
        #specificna baza trenera
        collection_name = f"coach_{coach_id}"
        return self.client.get_or_create_collection(name=collection_name)

    def add_to_collection(self, coach_id: str, ids: list, documents: list, metadatas: list, embeddings:list):
        collection = self.get_coach_collection(coach_id)
        existing = collection.get(ids=ids)
        new_ids = [i for i in ids if i not in existing['ids']]
        if not new_ids:
            print("Svi dokumenti već postoje u bazi.")
            return
        collection.add(
            ids=new_ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
    def query_collection(self, coach_id: str, query_embeddings: list, n_results: int = 3):
        collection = self.get_coach_collection(coach_id)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
    def delete_collection(self, coach_id: str):
        collection_name = f"coach_{coach_id}"
        try:
            self.client.delete_collection(name=collection_name)
            return True
        except Exception:
            return False
    
    def list_coaches(self):
        collections = self.client.list_collections()
        return [c.name.replace("coach_", "") for c in collections]