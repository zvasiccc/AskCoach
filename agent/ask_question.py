import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from flashrank import Ranker, RerankRequest
from prompts import SYSTEM_PROMPT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from printCoachDB import inspect_database
from ingest.embeddings import get_embeddings_model
from db.chroma import ChromaDBManager

load_dotenv()

MAX_DISTANCE = 0.8

embed_model = get_embeddings_model()
db = ChromaDBManager()
ranker = Ranker()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)



def ask_question(pitanje: str, coach_id: str) -> str:
    # 1. Embed pitanja
    query_vector = embed_model.embed_query(pitanje)


    collection = db.get_coach_collection(coach_id)
    #inspect_database(coach_id)
    if collection.count() == 0:
        return f"Trener '{coach_id}' nema unesene informacije u bazi."
    results = collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )

    raw_docs = results['documents'][0]
    
    passages = [
        {"id": i, "text": doc} for i, doc in enumerate(raw_docs)
    ]
    
    rerank_request = RerankRequest(query=pitanje, passages=passages)
    
    rerank_results = ranker.rerank(rerank_request)
    
    final_context_docs = [res['text'] for res in rerank_results[:3] if res['score'] > 0.5]
    
    if not final_context_docs:
        return "Nažalost, trener nije uneo tu informaciju."
    context = "\n---\n".join(final_context_docs)

    human_message = f"KONTEKST:\n{context}\n\nPITANJE:\n{pitanje}"


    response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_message)
        ])
    return response.content


if __name__ == "__main__":

    while True:
        user_query = input("Korisnik: ").strip()

        if not user_query:
            continue

        # llm_answer = ask_question(user_query, "trener_zeljko")
        llm_answer = ask_question(user_query, "trener_vukadin")
        print(f"\nASISTENT: {llm_answer}")
        print("-" * 50)