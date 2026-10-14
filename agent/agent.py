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
from shared.models import GroqModel, RoleEnum


from agent.retrieval import hybrid_retrieve
from prompts import get_system_prompt, NO_DOCS_MESSAGE, MAX_HISTORY_MESSAGES, CONDENSE_QUESTION_PROMPT
load_dotenv()

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")


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

def condense_question(question: str, history: list) -> str:
    if not history:
        return question

    history_text  = ""
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        if msg.role == "user":
            history_text  += f"Korisnik: {msg.content}\n"
        else:
            history_text  += f"Asistent: {msg.content}\n"

    prompt = CONDENSE_QUESTION_PROMPT.format(history=history_text , question=question)
    standalone = llm.invoke(prompt).content.strip()

    return standalone or question


def generate_answer_with_context_and_message_history(question: str, context: list[str], history: list | None = None, role: str = RoleEnum.Coach) -> str:
    history = history or []
    context_str = "\n---\n".join(context)
    system_prompt = get_system_prompt(role)

    messages = [SystemMessage(content=system_prompt)]
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    messages.append(HumanMessage(content=f"KONTEKST:\n{context_str}\n\nPITANJE:\n{question}"))
    return llm.invoke(messages).content

def ask_question(question: str, coach_id: str, client_id: str, history: list= [], role: str = RoleEnum.Coach):

    search_question = condense_question(question, history)

    raw_docs = hybrid_retrieve(search_question, coach_id, client_id)

    if not raw_docs:
        return NO_DOCS_MESSAGE, []

    reranked_documents = rerank(search_question, raw_docs)

    response = generate_answer_with_context_and_message_history(
        question=question, 
        context=reranked_documents, 
        history=history, 
        role=role
    )

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

