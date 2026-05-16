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


def ingest_raw_text(text_content, coach_id, client_id, source_name="manual_upload"):
    embeddings_model = get_embeddings_model()
    db = ChromaDBManager()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=350,
        separators=["\n\n", "\n", "!!! ", ". ", " "]
    )
    
    chunks = text_splitter.split_text(text_content)
    
    documents, embeddings, ids, metadatas = [], [], [], []


    for chunk in chunks:
        vector = embeddings_model.embeddings_query(chunk)
        
        documents.append(chunk)
        embeddings.append(vector)
        ids.append(str(uuid.uuid4()))
        metadatas.append({
            "source": source_name,
            "coach_id": coach_id,
            "client_id": client_id if client_id else "unknown"
            })

    db.add_to_collection(
        coach_id=coach_id,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"Uspesno dodato u bazu za trenera: {coach_id}")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
    
# if __name__ == "__main__":
#     ingest_raw_text(ingested_text_2, "trener_zeljko")