import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.embeddings import get_embeddings_model
from db.chroma import ChromaDBManager
from langchain_groq import ChatGroq

load_dotenv()

def ask_question(pitanje, coach_id):
    embed_model = get_embeddings_model()
    db = ChromaDBManager()
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

 
    query_vector = embed_model.embed_query(pitanje)
    # 3. Pretraga baze (Tražimo najsličniji dokument)
    collection = db.get_coach_collection(coach_id)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2  # Uzimamo top 2 najrelevantnija dela teksta
    )

    if not results['documents'][0] or results['distances'][0][0] > 1.4:
        return "Nažalost, trener nije uneo relevantne informacije."

    # Spajamo pronađene delove u jedan kontekst
    found_docs = results['documents'][0]
    context = "\n---\n".join(found_docs) if found_docs else "Nema relevantnih podataka u bazi."

    prompt = f"""
  
    Ako informacija nije u KONTEKSTU, odgovori sa: "Nažalost, nemam tu informaciju u bazi trenera".
    Odgovaraj u istom tonu u kom je prosledjeni kontekst, imitiraj stil govora prosledjenog konteksta.
    Tvoj govor je gramatički ispravan na srpskom jeziku (koristi padeže pravilno). 
    Izbegavaj doslovno prevođenje sa engleskog.
    Odgovaraj koncizno i precizno na postavljeno pitanje, bez nepotrebnih pojasnjenja.
    ZABRANJENO ti je da koristiš svoje opšte znanje. 
    Odgovori ISKLJUČIVO na osnovu dostavljenog KONTEKSTA.

    KONTEKST:
    {context}

    PITANJE:
    {pitanje}
    """
    response = llm.invoke(prompt)
    
    return response.content

if __name__ == "__main__":
    print("--- Fitness Coach AI (RAG Sistem) ---")
    print("Ukucaj 'exit' za kraj programa.\n")

    while True:
        user_query = input("Korisnik: ")

        if user_query.lower() in ["exit", "izlaz", "kraj"]:
            print("Gasim asistenta.")
            break
        
        if not user_query.strip():
            continue

        odgovor = ask_question(user_query, "trener_zeljko")
        
        print(f"\n ASISTENT: {odgovor}")
        print("-" * 50)