"""
Agent under test: agents/rag_agent.py
"""
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from agents.rag_agent import ask


model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")
faithfulness = FaithfulnessMetric(threshold=0.8, model=model)
contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=model)
contextual_recall = ContextualRecallMetric(threshold=0.7, model=model)
hallucination = HallucinationMetric(threshold=0.8, model=model)

booking_question = "What's the deadline for route changes on flights?"
unanswerable_question = "Am I allowed to make changes on my name in the reservation using kanjis?"

def get_test_case(question: str, expected_output=None) -> LLMTestCase:
    actual_output, retrieval_context = ask(input)
    test_case = LLMTestCase(
        input=question, 
        expected_output=expected_output, 
        retrieval_context=retrieval_context, 
        actual_output=actual_output)
    return test_case

def test_rag_answer_is_faithful_to_context():
    test_case = get_test_case(booking_question)
    evaluate(test_cases=[test_case], metrics=[faithfulness])

def test_rag_retrieval_is_precise_and_complete():
    test_case = get_test_case(booking_question)
    evaluate(test_cases=[test_case], metrics=[contextual_precision, contextual_recall])
def test_rag_no_hallucination_on_unanswerable_question():
    test_case = get_test_case(unanswerable_question, "I don't know, let me search information on that specific topic")
    evaluate(test_cases=[test_case], metrics=[hallucination])
