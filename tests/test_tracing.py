"""
DeepEval test skeleton: Component-Level Tracing (@observe)
Agent under test: agents/single_turn_agent.py

Unlike the other test files, which score ONE end-to-end LLMTestCase per
run, tracing lets you score INDIVIDUAL steps inside answer_query()
separately. single_turn_agent.py already wraps build_messages() and
call_llm() with @observe() -- this file evaluates those traced spans.
Docs: https://deepeval.com/docs/evaluation-component-level-llm-evals
      https://deepeval.com/docs/evaluation-llm-tracing
"""

import pytest
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from agents.single_turn_agent import answer_query


# 1. Open agents/single_turn_agent.py and look at how build_messages(),
#    call_llm(), and answer_query() are each wrapped with @observe(type=...).
#    call_llm is the one typed "llm" -- that's the span you'll score.


# 2. In single_turn_agent.py itself (not this file), give call_llm's
#    @observe() decorator a metrics=[...] argument (e.g. AnswerRelevancyMetric).
#    Then, inside call_llm, call
#    update_current_span(test_case=LLMTestCase(input=..., actual_output=...))
#    with the right values so that metric has something to score.


# 3. Back in this file: build a small list of Goldens (just an `input`
#    question each, no expected_output needed) and wrap them in an
#    EvaluationDataset


# 4. Write test_single_turn_agent_traced_components():
#    - Loop over dataset.evals_iterator()
#    - Call answer_query(golden.input) inside the loop so every @observe'd
#      span underneath it gets traced and scored
#    - Check DeepEval's docs for how the iterator expects results reported
#      back (e.g. dataset.evaluate(...) inside the loop)


# 5. (Stretch) Add a metrics=[...] to build_messages's @observe too (pick a
#    metric that makes sense for a prompt-building step, or write a custom
#    GEval one). Compare the per-span results to the single end-to-end
#    LLMTestCase you built in test_single_turn_agent.py -- tracing should
#    tell you WHICH component drove a low score, not just that the final
#    answer scored low.
