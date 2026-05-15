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

from prompts import QUERY_EXPANSION_PROMPT, SYSTEM_PROMPT, STOP_WORDS, MAX_DISTANCE
from ingest.embeddings import get_embeddings_model
from db.chroma import ChromaDBManager


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
db = ChromaDBManager() 
embeddings_model = get_embeddings_model()

def tokenize(text: str) -> list[str]:
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

def expand_query(pitanje: str) -> list[str]:
    try:
        prompt = QUERY_EXPANSION_PROMPT.format(pitanje=pitanje)
        response = llm.invoke(prompt)
        question_variants = json.loads(response.content)
        return [pitanje] + question_variants
    except Exception:
        return [pitanje]


def bm25_search(pitanje: str, bm25, all_docs: list[str], top_k: int = 10) -> list[tuple[str, float]]:
    scores = bm25.get_scores(tokenize(pitanje))
    doc_scores = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
    return [(doc, score) for doc, score in doc_scores[:top_k] if score > 0]

#vraca bm25 indeks svih chunkova trenera
def get_bm25(collection, coach_id: str, db_cache):
    count = collection.count()
    if coach_id not in db_cache or db_cache[coach_id]["count"] != count:
        all_docs = collection.get()["documents"]
        tokenized = [tokenize(doc) for doc in all_docs]
        db_cache[coach_id] = {
            "bm25": BM25Okapi(tokenized),
            "docs": all_docs,
            "count": count
        }
    return db_cache[coach_id]["bm25"], db_cache[coach_id]["docs"]

def vector_search(question_variants: list[str], collection, n_results: int = 10) -> dict[str, float]:
    seen_docs = {}
    for variant in question_variants:
        query_vector = embeddings_model.embeddings_query(variant)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if doc not in seen_docs or dist < seen_docs[doc]:
                seen_docs[doc] = dist
    return {doc: dist for doc, dist in seen_docs.items() if dist <= MAX_DISTANCE}


def hybrid_retrieve(question: str, coach_id: str) -> list[str]:
    collection = db.get_coach_collection(coach_id)

    if collection.count() == 0:
        return []
    
    expand_query_variants = expand_query(question)
    bm25_obj, all_docs = get_bm25(collection, coach_id, db._bm25_cache)

    if not all_docs:
        return []

    top_bm25_docs = bm25_search(question, bm25_obj, all_docs)
    top_vector_docs = vector_search(expand_query_variants, collection)

    if not top_vector_docs and not top_bm25_docs:
        return []

    K = 60
    rrf_scores = {}

    for rank, (doc, _) in enumerate(sorted(top_vector_docs.items(), key=lambda x: x[1]), start=1):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (K + rank)

    for rank, (doc, _) in enumerate(top_bm25_docs, start=1):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (K + rank)

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:10]]


