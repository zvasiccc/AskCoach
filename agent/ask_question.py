import json
import os
import re
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from flashrank import Ranker, RerankRequest
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from rank_bm25 import BM25Okapi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompts import SYSTEM_PROMPT
from ingest.embeddings import get_embeddings_model
from db.chroma import ChromaDBManager

load_dotenv()

MAX_DISTANCE = 1.1
_bm25_cache = {}

embed_model = get_embeddings_model()
db = ChromaDBManager()
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

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

STOP_WORDS = {
    "da", "li", "se", "je", "su", "i", "u", "na", "za", "bi", "sam",
    "sto", "kako", "koliko", "koji", "koja", "koje", "ako", "ili",
    "ali", "jer", "što", "sve", "ovo", "ono", "taj", "ta", "to",
    "mi", "vi", "oni", "one", "moj", "tvoj", "svoj", "neki", "može",
    "treba", "imam", "ima", "biti", "bih", "moze", "trebam", "hocu"
}

QUERY_EXPANSION_PROMPT = """Ti si ekspert za fitnes i treniranje.
Dato ti je korisnikovo pitanje. Generiši 2 različite varijante tog pitanja koje imaju IDENTIČNO ZNAČENJE,
ali su formulisane drugačije.

Vrati SAMO JSON listu, bez ikakvog teksta pre ili posle. Primer formata:
["varijanta 1", "varijanta 2"]

Pitanje: {pitanje}"""


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


def tokenize(text: str) -> list[str]:
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def get_bm25(collection, coach_id: str):
    """BM25 keš — pravi se jednom, osvežava se samo kad se baza promeni."""
    count = collection.count()
    if coach_id not in _bm25_cache or _bm25_cache[coach_id]["count"] != count:
        all_docs = collection.get()["documents"]
        tokenized = [tokenize(doc) for doc in all_docs]
        _bm25_cache[coach_id] = {
            "bm25": BM25Okapi(tokenized),
            "docs": all_docs,
            "count": count
        }
    return _bm25_cache[coach_id]["bm25"], _bm25_cache[coach_id]["docs"]


def expand_query(pitanje: str) -> list[str]:
    try:
        prompt = QUERY_EXPANSION_PROMPT.format(pitanje=pitanje)
        response = llm.invoke(prompt)
        varijante = json.loads(response.content)
        return [pitanje] + varijante
    except Exception:
        return [pitanje]


def vector_search(varijante: list[str], collection, n_results: int = 10) -> dict[str, float]:
    seen_docs = {}
    for varijanta in varijante:
        query_vector = embed_model.embed_query(varijanta)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if doc not in seen_docs or dist < seen_docs[doc]:
                seen_docs[doc] = dist
    return {doc: dist for doc, dist in seen_docs.items() if dist <= MAX_DISTANCE}


def bm25_search(pitanje: str, bm25, all_docs: list[str], top_k: int = 10) -> list[tuple[str, float]]:
    scores = bm25.get_scores(tokenize(pitanje))
    doc_scores = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
    return [(doc, score) for doc, score in doc_scores[:top_k] if score > 0]


def hybrid_retrieve(pitanje: str, collection, coach_id: str) -> list[str]:
    varijante = expand_query(pitanje)
    bm25, all_docs = get_bm25(collection, coach_id)

    if not all_docs:
        return []

    vector_docs = vector_search(varijante, collection)
    bm25_docs = bm25_search(pitanje, bm25, all_docs)

    if not vector_docs and not bm25_docs:
        return []

    K = 60
    rrf_scores = {}

    for rank, (doc, _) in enumerate(sorted(vector_docs.items(), key=lambda x: x[1]), start=1):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (K + rank)

    for rank, (doc, _) in enumerate(bm25_docs, start=1):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (K + rank)

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:10]]


def ask_question(pitanje: str, coach_id: str,history:list=[]):
    collection = db.get_coach_collection(coach_id)

    if collection.count() == 0:
        return f"Trener '{coach_id}' nema unesene informacije u bazi.", []

    raw_docs = hybrid_retrieve(pitanje, collection, coach_id)

    if not raw_docs:
        return "Nažalost, trener nije uneo tu informaciju.", []

    passages = [{"id": i, "text": doc} for i, doc in enumerate(raw_docs)]
    rerank_results = ranker.rerank(RerankRequest(query=pitanje, passages=passages))
    final_context_docs = [res["text"] for res in rerank_results[:3]]

    context_str = "\n---\n".join(final_context_docs)
    print(f"DEBUG context:\n{context_str}")
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    # Dodajemo trenutno pitanje sa kontekstom
    messages.append(HumanMessage(content=f"KONTEKST:\n{context_str}\n\nPITANJE:\n{pitanje}"))

    response = llm.invoke(messages)

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
    print(f"Faithfulness (Vernost):   {faithfulness.score:.2f}")
    print(f"Relevancy (Relevantnost): {relevancy.score:.2f}")
    print(f"Razlog:                   {faithfulness.reason}")


if __name__ == "__main__":
    print("--- Fitness Coach AI (Hybrid RAG) ---")
    eval_mode = input("Evaluacioni mod? (y/n): ").strip().lower() == "y"

    while True:
        user_query = input("\nKorisnik: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "izlaz"]:
            break

        llm_answer, used_context = ask_question(user_query, "trener_nikola")
        print(f"\nASISTENT: {llm_answer}")

        if eval_mode and used_context:
            try:
                print("\n[EVALUACIJA U TOKU...]")
                run_evaluation(user_query, llm_answer, used_context)
            except Exception as e:
                print(f"Evaluacija nije uspela: {e}")

        print("-" * 50)