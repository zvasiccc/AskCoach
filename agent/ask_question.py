import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from flashrank import Ranker, RerankRequest
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

from prompts import SYSTEM_PROMPT
from ingest.embeddings import get_embeddings_model
from db.chroma import ChromaDBManager

load_dotenv()

embed_model = get_embeddings_model()
db = ChromaDBManager()
ranker = Ranker()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

eval_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = chat_model.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Llama 3.1 8B via Groq"

eval_model = GroqModel(model=eval_llm) 

def ask_question(pitanje: str, coach_id: str):
    # 1. Embed pitanja i Retrieval
    query_vector = embed_model.embed_query(pitanje)
    collection = db.get_coach_collection(coach_id)
    
    if collection.count() == 0:
        return f"Trener '{coach_id}' nema unesene informacije u bazi.", []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=10
    )

    raw_docs = results['documents'][0]
    
    # 2. Reranking (Advanced RAG korak)
    passages = [{"id": i, "text": doc} for i, doc in enumerate(raw_docs)]
    rerank_request = RerankRequest(query=pitanje, passages=passages)
    rerank_results = ranker.rerank(rerank_request)
    
    # Filtriranje po skoru
    final_context_docs = [res['text'] for res in rerank_results[:3] if res['score'] > 0.3]
    
    if not final_context_docs:
        return "Nažalost, trener nije uneo tu informaciju.", []

    context_str = "\n---\n".join(final_context_docs)

    # 3. Generisanje odgovora (LLM)
    human_message = f"KONTEKST:\n{context_str}\n\nPITANJE:\n{pitanje}"
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_message)
    ])
    
    # Vraćamo odgovor i listu dokumenata (zbog evaluacije)
    return response.content, final_context_docs

def run_evaluation(pitanje, odgovor, kontekst):

    test_case = LLMTestCase(
        input=pitanje,
        actual_output=odgovor,
        retrieval_context=kontekst
    )
    
    faithfulness = FaithfulnessMetric(threshold=0.7, model=eval_model)
    relevancy = AnswerRelevancyMetric(threshold=0.5, model=eval_model)
    
    faithfulness.measure(test_case)
    relevancy.measure(test_case)
    
    print(f"Faithfulness (Vernost): {faithfulness.score:.2f}")
    print(f"Relevancy (Relevantnost): {relevancy.score:.2f}")
    print(f"Razlog za relevantnost: {faithfulness.reason}")

if __name__ == "__main__":
    print("--- Fitness Coach AI (RAG + DeepEval) ---")

    while True:
        user_query = input("\nKorisnik: ").strip()

        if user_query.lower() in ["exit", "izlaz"]:
            break

        # Dobijamo odgovor i kontekst
        llm_answer, used_context = ask_question(user_query, "trener_milos")
        
        print(f"\nASISTENT: {llm_answer}")
        
        # Ako imamo kontekst, radimo evaluaciju
        if used_context:
            try:
                run_evaluation(user_query, llm_answer, used_context)
            except Exception as e:
                print(f"Evaluacija nije uspela: {e}")
        
        print("-" * 50)