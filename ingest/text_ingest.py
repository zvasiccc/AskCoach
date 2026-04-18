import os
import sys
import uuid
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir)) 
sys.path.append(current_dir) 

from embeddings import get_embeddings_model
from db.chroma import ChromaDBManager

load_dotenv()

def ingest_raw_text(text_content, coach_id, source_name="manual_upload"):
    embed_model = get_embeddings_model()
    db = ChromaDBManager()

    chunks = [text_content[i:i+500] for i in range(0, len(text_content), 500)]
    
    documents, embeddings, ids, metadatas = [], [], [], []

    print(f"Obrađujem {len(chunks)} delova teksta...")

    for chunk in chunks:
        vector = embed_model.embed_query(chunk)
        
        documents.append(chunk)
        embeddings.append(vector)
        ids.append(str(uuid.uuid4()))
        metadatas.append({"source": source_name, "coach_id": coach_id})

    db.add_to_collection(
        coach_id=coach_id,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"✅ Uspešno dodato u bazu za trenera: {coach_id}")

if __name__ == "__main__":
    moj_tekst = "Zgibovi su ključna vežba za razvoj leđnih mišića. Preporucujem svima 3 serije od 8 do 12 ponavljanja."
    ingest_raw_text(moj_tekst, "trener_zeljko")