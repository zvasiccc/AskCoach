import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.embeddings import get_embeddings_model
from db.chroma import ChromaDBManager

embed_model = get_embeddings_model()
db = ChromaDBManager()

test_pitanja = [
    # Pitanja koja TREBA da nađu odgovor
    "koliko serija zgibova",
    "treneru koliko serija zgibova da radim",
    "koliko serija zgibova ako sam juce preskocio trening",
    "kako da radim zgibove",
    "vežbe za leđa",
    "koliko serija propadanja",
    "koliko serija sklekova",
    "koliko serija sklekova ako sam juce preskocio trening",
    "koliko serija u celom treningu da radim",
    "koje vezbe da radim za donji deo tela",
    "vezbe za grudi",
    "kako da smršam",
    "recept za palačinke",
    "vreme u beogradu",
]

collection = db.get_coach_collection("trener_zeljko")

print(f"{'PITANJE':<35} | {'DISTANCA':<10} | {'DOKUMENT (prva 60 znaka)'}")
print("-" * 100)

for pitanje in test_pitanja:
    vector = embed_model.embed_query(pitanje)
    results = collection.query(query_embeddings=[vector], n_results=1)
    
    dist = results['distances'][0][0]
    doc = results['documents'][0][0][:60].replace('\n', ' ')
    
    print(f"{pitanje:<35} | {dist:<10.4f} | {doc}")