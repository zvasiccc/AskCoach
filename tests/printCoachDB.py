import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.chroma import ChromaDBManager

def inspect_database(coach_id):
    db_manager = ChromaDBManager()
    collection = db_manager.get_coach_collection(coach_id)
    
    results = collection.get()
    
    print(f"\npregled baze za {coach_id}")
    print(f"Ukupno isecaka u bazi: {len(results['ids'])}\n")
    
    for i in range(len(results['ids'])):
        print(f"ID: {results['ids'][i]}")
        print(f"TEKST: {results['documents'][i][:150]}...")
        print(f"METADATA: {results['metadatas'][i]}")
        print("-" * 30)
