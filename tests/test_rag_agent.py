"""
DeepEval test skeleton: RAG Agent
Agent under test: agents/rag_agent.py

RAG-specific metrics need `retrieval_context` (what was retrieved) in
addition to input/actual_output/expected_output.
Docs: https://docs.confident-ai.com/docs/metrics-contextual-precision
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from agents.rag_agent import ask


# 1. Write a helper get_test_case(question, expected_output=None) that
#    calls ask() (check what it returns in rag_agent.py) and builds an
#    LLMTestCase with input, actual_output, retrieval_context, and
#    expected_output


# 2. Write test_rag_answer_is_faithful_to_context():
#    - Pick a question you know data/knowledge_base/ can answer
#    - Build the test case
#    - Assert with FaithfulnessMetric that the answer doesn't contradict
#      the retrieved context


# 3. Write test_rag_retrieval_is_precise_and_complete():
#    - Reuse/build a test case with a known expected_output
#    - Assert with ContextualPrecisionMetric AND ContextualRecallMetric
#      (hint: assert_test takes a list of metrics)


# 4. Write test_rag_no_hallucination_on_unanswerable_question():
#    - Ask a question the knowledge base can NOT answer
#    - Decide what the agent SHOULD do (say "I don't know" vs guessing)
#    - Assert with HallucinationMetric using retrieval_context as context


# 5. (Stretch) Parametrize step 2 across each .txt file in
#    data/knowledge_base/ with one question per document
