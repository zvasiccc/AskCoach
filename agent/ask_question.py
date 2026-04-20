import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from printCoachDB import inspect_database
from ingest.embeddings import get_embeddings_model
from db.chroma import ChromaDBManager

load_dotenv()

MAX_DISTANCE = 0.8

embed_model = get_embeddings_model()
db = ChromaDBManager()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

SYSTEM_PROMPT = """Ti si asistent koji odgovara ISKLJUČIVO na osnovu dostavljenog KONTEKSTA.

PRAVILA (ne smeš ih prekršiti):
- Ako odgovor nije u KONTEKSTU, odgovori tačno ovako: "Nažalost, trener nije uneo tu informaciju."
- ZABRANJENO ti je da koristiš svoje opšte znanje, pretpostavke ili zaključivanje van konteksta.
- ZABRANJENO ti je da izmišljaš, dopunjuješ ili pretpostavljaš informacije.
- Ako KONTEKST ne pominje DIREKTNO temu iz pitanja, odgovori: "Nažalost, trener nije uneo tu informaciju."
- NE pravi analogije između vežbi. Ako je pitanje o vežbi X, a kontekst govori o vežbi Y — to nije relevantan odgovor
- Odgovaraj u istom tonu i stilu govora kao što je napisan KONTEKST.
- Govor mora biti gramatički ispravan na srpskom jeziku (koristiti padeže pravilno).
- Izbegavaj doslovno prevođenje sa engleskog.
- Odgovaraj koncizno i precizno, bez nepotrebnih pojašnjenja."""

def ask_question(pitanje: str, coach_id: str) -> str:
    # 1. Embed pitanja
    query_vector = embed_model.embed_query(pitanje)

    # 2. Pretraga vektorske baze
    try:
        collection = db.get_coach_collection(coach_id)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        print(f"DEBUG documents: {results['documents']}")
        print(f"DEBUG distances: {results['distances']}")
    except Exception as e:
        return f"Greška pri pretrazi baze: {e}"

    # 3. Provera da li ima rezultata i da li su dovoljno relevantni
    documents = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]


    inspect_database(coach_id)


    if not documents:
        return "Nažalost, trener nije uneo nikakve informacije."

    relevantni = [
        doc for doc, dist in zip(documents, distances)
        if dist <= MAX_DISTANCE
    ]

    if not relevantni:
        return "Nažalost, trener nije uneo tu informaciju."

    # 4. Spajamo relevantne delove u kontekst
    context = "\n---\n".join(relevantni)

    # 5. LLM poziv
    human_message = f"KONTEKST:\n{context}\n\nPITANJE:\n{pitanje}"

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_message)
        ])
        return response.content
    except Exception as e:
        return f"Greška pri generisanju odgovora: {e}"


if __name__ == "__main__":
    print("--- Fitness Coach AI (RAG Sistem) ---")
    print("Ukucaj 'exit' za kraj programa.\n")

    while True:
        user_query = input("Korisnik: ").strip()

        if not user_query:
            continue

        if user_query.lower() in ["exit", "izlaz", "kraj"]:
            print("Gasim asistenta.")
            break

        odgovor = ask_question(user_query, "trener_zeljko")
        print(f"\nASISTENT: {odgovor}")
        print("-" * 50)