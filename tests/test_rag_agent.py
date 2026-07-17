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

# Runs the RAG agent and builds an LLMTestCase. `context` is the ground
# truth (independent of what got retrieved); `retrieval_context` is whatever
# the agent's TF-IDF search actually pulled for this question.
def get_test_case(question: str, expected_output=None) -> LLMTestCase:
    result = ask(question)
    test_case = LLMTestCase(
        input=question, 
        expected_output=expected_output, 
        retrieval_context=result["retrieval_context"],
        context=["Trailhead Travel does not support name changes using non-Latin scripts; only Latin-alphabet corrections up to 3 characters are permitted."], 
        actual_output=result["answer"]),
    return test_case

# A question the knowledge base can answer -- checks the answer doesn't
# contradict what was actually retrieved.
def test_rag_answer_is_faithful_to_context():
    test_case = get_test_case(booking_question, "3 hours prior the reserved flight")
    evaluate(test_cases=[test_case], metrics=[faithfulness])

# Same question -- checks the retriever itself, not the generated answer:
# did it pull the right docs (precision) and not miss relevant ones (recall)?
def test_rag_retrieval_is_precise_and_complete():
    test_case = get_test_case(booking_question, "3 hours prior the reserved flight")
    evaluate(test_cases=[test_case], metrics=[contextual_precision, contextual_recall])

# A question the knowledge base can't actually answer -- checks the agent
# doesn't invent an answer beyond what's true (per `context`).
def test_rag_no_hallucination_on_unanswerable_question():
    test_case = get_test_case(unanswerable_question, "I don't know, let me search information on that specific topic")
    evaluate(test_cases=[test_case], metrics=[hallucination])
