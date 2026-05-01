import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir)) 
sys.path.append(current_dir) 

from embeddings import get_embeddings_model
from db.chroma import ChromaDBManager

load_dotenv()

ingestd_text = """1. OSNOVNI PRINCIPI TRENINGA
Svaki trening mora početi zagrevanjem od 10 minuta (lagani džoging ili vijača) i dinamičkim istezanjem. Odmor između serija treba da bude 60-90 sekundi. Fokus je uvek na formi, a ne na težini.

2. VEŽBE ZA GORNJI DEO TELA

Zgibovi (Pull-ups): Ključna vežba za razvoj leđnih mišića (latissimus dorsi). Preporučujem 3 serije od 8 do 12 ponavljanja. Ako ne možete da uradite pun zgib, koristite elastične trake za asistenciju.

Sklekovi (Push-ups): Osnovna vežba za grudi i triceps. Raditi 4 serije do otkaza. Vodite računa da laktovi budu uz telo pod uglom od 45 stepeni.

Military Press: Za razvoj ramena. Raditi 3 serije po 10 ponavljanja sa dvoručnim tegom ili bučicama.

3. VEŽBE ZA DONJI DEO TELA

Čučnjevi (Squats): Kraljica svih vežbi. Fokus na dubinu i prava leđa. Raditi 4 serije od 10 do 15 ponavljanja.

Iskorak (Lunges): Odlična vežba za stabilnost i gluteus. Raditi 3 serije od 12 ponavljanja po nozi.

Mrtvo dizanje (Deadlift): Za zadnju ložu i donji deo leđa. Raditi oprezno, 3 serije po 8 ponavljanja sa fokusom na neutralnu kičmu.

4. ISHRANA I SUPLEMENTACIJA

Proteini: Svaki obrok treba da sadrži izvor proteina (piletina, riba, jaja, posni sir). Ciljati 1.8g do 2g proteina po kilogramu telesne težine.

Hidrati: Glavni izvor energije. Fokusirati se na složene hidrate poput pirinča, ovsenih pahuljica i batata.

Suplementi: Preporučujem Kreatin Monohidrat (5g dnevno) i Whey protein nakon treninga radi bržeg oporavka.

5. OPORAVAK I BOLEST
San je podjednako važan kao i trening. Spavati najmanje 7-8 sati. U slučaju bolesti (povišena temperatura, malaksalost), odmah prekinuti trening. Piti dosta čajeva i vode, jesti lagane supe i krekere dok se organizam ne oporavi. Ne vraćati se u teretanu dok simptomi potpuno ne nestanu."""
ingested_text_2="radi sklekove po 3 serije uvek."
def ingest_raw_text(text_content, coach_id, source_name="manual_upload"):
    embed_model = get_embeddings_model()
    db = ChromaDBManager()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = text_splitter.split_text(text_content)
    
    documents, embeddings, ids, metadatas = [], [], [], []


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
    print(f"Uspesno dodato u bazu za trenera: {coach_id}")

# if __name__ == "__main__":
#     ingest_raw_text(ingested_text_2, "trener_milos")