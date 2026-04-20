import os
import sys
from dotenv import load_dotenv

# Podešavanje putanja da vidi tvoje module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.chroma import ChromaDBManager

def inspect_database(coach_id):
    db_manager = ChromaDBManager()
    collection = db_manager.get_coach_collection(coach_id)
    
    # .get() bez argumenata vraća SVE iz kolekcije
    results = collection.get()
    
    print(f"\n--- PREGLED BAZE ZA: {coach_id} ---")
    print(f"Ukupno isecaka (chunks) u bazi: {len(results['ids'])}\n")
    
    for i in range(len(results['ids'])):
        print(f"🆔 ID: {results['ids'][i]}")
        print(f"📄 TEKST: {results['documents'][i][:150]}...") # Ispisujemo prvih 150 karaktera
        print(f"🏷️ METADATA: {results['metadatas'][i]}")
        print("-" * 30)
