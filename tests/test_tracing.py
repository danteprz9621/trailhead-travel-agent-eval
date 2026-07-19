"""
Component-Level Tracing (@observe)
"""

import pytest
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel

from agents.single_turn_agent import answer_query

model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=model)

goldens = [
    Golden(input="What's the baggage policy?"),
    Golden(input="What's the refund policy?"),
    Golden(input="Is there a loyalty program?")]

dataset = EvaluationDataset(goldens=goldens)

# Drives answer_query() through evals_iterator so every @observe'd span
# underneath it gets scored -- prompt_alignment on build_messages and
# answer_relevancy on call_llm (both set in single_turn_agent.py), plus
# answer_relevancy again here at the end-to-end level.
def test_single_turn_agent_traced_components():
    for golden in dataset.evals_iterator(metrics=[answer_relevancy]):
        answer_query(golden.input)
