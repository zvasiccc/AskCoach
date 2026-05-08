# evaluation/run_eval.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from datasets import Dataset
from agent.ask_question import ask_question
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

groq_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
))

metrics = [
    Faithfulness(llm=groq_llm),
    AnswerRelevancy(llm=groq_llm),
    ContextPrecision(llm=groq_llm),
    ContextRecall(llm=groq_llm),
]

def run_evaluation(coach_id: str, test_cases: list):
    questions, answers, contexts, ground_truths = [], [], [], []

    for case in test_cases:
        question = case["question"]
        
        # Dohvati odgovor
        answer, reranked_docs = ask_question(question, coach_id)
        
        questions.append(question)
        answers.append(answer)
        contexts.append(reranked_docs)
        ground_truths.append(case["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(dataset,metrics=metrics,llm=groq_llm)
    print(f"Faithfulness:      {results['faithfulness']:.3f}  (1.0 = nikad ne izmišlja)")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}  (1.0 = uvek odgovara na pitanje)")
    print(f"Context Precision: {results['context_precision']:.3f}  (1.0 = samo relevantni chunkovi)")
    print(f"Context Recall:    {results['context_recall']:.3f}  (1.0 = uvek nalazi pravi chunk)")
    
    return results

if __name__ == "__main__":
    from evaluation.test_dataset import test_cases
    run_evaluation("trener_nikola", test_cases)