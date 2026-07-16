"""
DeepEval test skeleton: Single-Turn Agent
Agent under test: agents/single_turn_agent.py

Follow the numbered comments top to bottom. Each comment describes ONE
thing to implement below it -- fill in the code yourself.
Docs: https://docs.confident-ai.com/docs/getting-started
"""

import pytest
import deepeval
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.single_turn_agent import answer_query
deepeval.config.throughputs = 1

# 1. Create an AnswerRelevancyMetric instance with a threshold of 0.7
model = GeminiModel("gemini-2.5-flash")
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=model)

# 2. Create a custom GEval metric called "Correctness" that checks whether
#    the actual_output correctly answers the input, using
#    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

correctness = GEval(
    name="correctness",
    criteria = ("Determine whether the actual output conveys the same factual information as expected output."),
    model = model,
    threshold = 0.8,
    evaluation_params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ]
)


# 3. Write test_single_turn_answers_relevantly():
#    - Define a sample travel question for the agent
#    - Call answer_query() to get the actual_output
#    - Build an LLMTestCase with input and actual_output
#    - Run assert_test() with both metrics from steps 1 and 2

def test_single_turn_answers_relevantly():
    input = "What's the flight cancellation policy?"
    actual_output = answer_query(input)
    test_case = LLMTestCase(
        input = input,
        actual_output=actual_output
    )
    evaluate(test_cases=[test_case], metrics=[correctness, answer_relevancy])


# 4. Write test_single_turn_handles_out_of_scope_question():
#    - Ask something the agent has no business answering
#      (e.g. "What's the capital of France?")
#    - Decide what "success" means here and pick/build the right metric
#    - Assert on it
def test_single_turn_handles_out_of_scope_question():
    input = "When's mexican independence day?"
    actual_output = answer_query(input)
    test_case = LLMTestCase(
        input = input,
        actual_output=actual_output
    )
    evaluate(test_cases=[test_case], metrics=[answer_relevancy])


# 5. Parametrize test_single_turn_answers_relevantly with
#    pytest.mark.parametrize so it runs over a small list of sample
#    questions instead of one hardcoded one

@pytest.mark.parametrize("input", 
                         ["What's the flight cancellation policy?",
                          "What's the baggage policy?",
                          "What's the refund policy?"])
def test_single_turn_answers_relevantly(input):
    actual_output = answer_query(input)
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output
    )
    evaluate(test_cases=[test_case], metrics=[answer_relevancy])