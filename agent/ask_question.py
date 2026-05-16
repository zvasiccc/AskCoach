import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from flashrank import Ranker, RerankRequest
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest.embeddings import get_embeddings_model


from agent.retrieval import hybrid_retrieve
from prompts import  MAX_DISTANCE, get_system_prompt
load_dotenv()

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
embeddings_model = get_embeddings_model()

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
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.load_model().ainvoke(prompt)).content

    def get_model_name(self):
        return self.model.model_name


eval_model = GroqModel(model=eval_llm)


def vector_search(varijante: list[str], collection, n_results: int = 10) -> dict[str, float]:
    seen_docs = {}
    for varijanta in varijante:
        query_vector = embeddings_model.embeddings_query(varijanta)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if doc not in seen_docs or dist < seen_docs[doc]:
                seen_docs[doc] = dist
    return {doc: dist for doc, dist in seen_docs.items() if dist <= MAX_DISTANCE}



def rerank(question: str, docs: list[str], top_k: int = 3) -> list[str]:
    if not docs:
        return []

    passages = [{"id": i, "text": doc} for i, doc in enumerate(docs)]
    results = ranker.rerank(RerankRequest(query=question, passages=passages))
    return [res["text"] for res in results[:top_k]]

def generate_promt_with_context_and_message_history(question: str, context: list[str], history: list = [], role: str = "trener") -> str:
    context_str = "\n---\n".join(context)
    system_prompt = get_system_prompt(role)

    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    messages.append(HumanMessage(content=f"KONTEKST:\n{context_str}\n\nPITANJE:\n{question}"))
    return llm.invoke(messages).content

def ask_question(question: str, coach_id: str,client_id:str, history:list=[],role: str = "trener"):

    raw_docs = hybrid_retrieve(question, coach_id,client_id)

    if not raw_docs :
        return "Nazalost, trazena informacija se ne nalazi u bazi znanja.", []

    reranked_documents = rerank(question, raw_docs)

    response = generate_promt_with_context_and_message_history(question, reranked_documents, history,role)

    return response, reranked_documents


def run_evaluation(question, answer, context):
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context
    )
    faithfulness = FaithfulnessMetric(threshold=0.7, model=eval_model)
    relevancy = AnswerRelevancyMetric(threshold=0.5, model=eval_model)
    faithfulness.measure(test_case)
    relevancy.measure(test_case)
    print(f"Faithfulness:{faithfulness.score:.2f}")
    print(f"Relevancy:{relevancy.score:.2f}")
    print(f"Razlog:{faithfulness.reason}")


# if __name__ == "__main__":
#     eval_mode = input("Evaluacioni mod? (y/n): ").strip().lower() == "y"

#     while True:
#         user_query = input("\nKorisnik: ").strip()
#         if not user_query:
#             continue
#         if user_query.lower() in ["exit", "izlaz"]:
#             break

#         llm_answer, used_context = ask_question(user_query, "trener_nikola")
#         print(f"\nASISTENT: {llm_answer}")

#         if eval_mode and used_context:
#             try:
#                 print("\n[EVALUACIJA U TOKU...]")
#                 run_evaluation(user_query, llm_answer, used_context)
#             except Exception as e:
#                 print(f"Evaluacija nije uspela: {e}")

#         print("-" * 50)