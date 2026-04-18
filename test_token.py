import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

# Ova linija učitava podatke iz .env fajla u memoriju
load_dotenv()

def test_connection():
    token = os.getenv("HF_TOKEN")
    print(f"DEBUG: Token počinje sa: {token[:5]}...") # Provera da li je učitan
    
    try:
        embed_model = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=token
        )
        test_vector = embed_model.embed_query("Provera API konekcije")
        print(f"Uspeh! Dužina vektora: {len(test_vector)}")
    except Exception as e:
        print(f"Greška: {e}")

if __name__ == "__main__":
    test_connection()