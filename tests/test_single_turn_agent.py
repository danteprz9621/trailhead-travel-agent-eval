"""
Agent under test: agents/single_turn_agent.py
"""

import pytest
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from agents.single_turn_agent import answer_query

model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=model)

# AnswerRelevancy only checks that the reply is on-topic, not that it's
# factually right -- this custom GEval metric covers the "is it actually
# correct" half of the picture.
correctness = GEval(
    name="correctness",
    criteria = ("Determine whether the actual output conveys the same factual information as expected output."),
    model = model,
    threshold = 0.8,
    evaluation_params = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT
    ]
)


# Golden path: a clearly in-scope travel question should score well on both
# relevancy and factual correctness.
def test_single_turn_answers_relevantly():
    input = "What's the flight cancellation policy?"
    actual_output = answer_query(input)
    test_case = LLMTestCase(
        input = input,
        actual_output=actual_output
    )
    evaluate(test_cases=[test_case], metrics=[correctness, answer_relevancy])


# Edge case: a question outside the agent's travel-support scope. Only
# checked for relevancy -- there's no single "correct" factual answer to
# grade here, just whether the reply stays on-topic for what was asked.
def test_single_turn_handles_out_of_scope_question():
    input = "When's mexican independence day?"
    actual_output = answer_query(input)
    test_case = LLMTestCase(
        input = input,
        actual_output=actual_output
    )
    evaluate(test_cases=[test_case], metrics=[answer_relevancy])


# Same relevancy check as the golden-path test above, run across a small
# batch of in-scope questions instead of just one.
@pytest.mark.parametrize("input",
                         ["What's the flight cancellation policy?",
                          "What's the baggage policy?",
                          "What's the refund policy?"])
def test_single_turn_answers_relevantly_param(input):
    actual_output = answer_query(input)
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output
    )
    evaluate(test_cases=[test_case], metrics=[answer_relevancy])