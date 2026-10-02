import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from flashrank import Ranker, RerankRequest
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest.embeddings import get_embeddings_model
from shared.models import GroqModel, RoleEnum


from agent.retrieval import hybrid_retrieve
from prompts import  MAX_DISTANCE, get_system_prompt
load_dotenv()

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
embeddings_model = get_embeddings_model()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

eval_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

eval_model = GroqModel(model=eval_llm)




def rerank(question: str, docs: list[str], top_k: int = 3) -> list[str]:
    if not docs:
        return []

    passages = [{"id": i, "text": doc} for i, doc in enumerate(docs)]
    results = ranker.rerank(RerankRequest(query=question, passages=passages))
    return [res["text"] for res in results[:top_k]]

def generate_promt_with_context_and_message_history(question: str, context: list[str], history: list = [], role: str = RoleEnum.Coach) -> str:
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

def ask_question(question: str, coach_id: str,client_id:str, history:list=[],role: str = RoleEnum.Coach):

    raw_docs = hybrid_retrieve(question, coach_id,client_id)

    if not raw_docs :
        return "Nazalost, trazena informacija se ne nalazi u bazi znanja.", []

    reranked_documents = rerank(question, raw_docs)

    response =  (question, reranked_documents, history,role)

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


if __name__ == "__main__":

    question = "Koja je osnovu suplementacije za sve svoje klijente?"

    user_id_1 = "korisnik_jovan"
    answer_jovan, documents_jovan = ask_question(question, coach_id=user_id_1, client_id=None)

    user_id_2 = "korisnik_stefan"
    answer_stefan, documents_stefan = ask_question(question, coach_id=user_id_2, client_id=None)

    print(f"Kolekcija dokumenata za korisnika: {user_id_1}")
    print(f"Dohvaceni dokumenti:\n{documents_jovan}\n")
    print(f"Odgovor asistenta:\n{answer_jovan}\n")

    print("--------------------------------------------------------")

    print(f"Kolekcija dokumenata za korisnika: {user_id_2}")
    print(f"Dohvaceni dokumenti:\n{documents_stefan}\n")
    print(f"Odgovor asistenta:\n{answer_stefan}\n")