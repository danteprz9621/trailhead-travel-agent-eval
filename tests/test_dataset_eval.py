"""
DeepEval test skeleton: Dataset-based Evaluation & Synthetic Data
Ties everything together: build a golden dataset, evaluate many test
cases at once, and (optionally) generate more with the Synthesizer.

Docs: https://docs.confident-ai.com/docs/evaluation-datasets
      https://docs.confident-ai.com/docs/synthesizer-introduction
"""

import pytest
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from agents.single_turn_agent import answer_query
from agents.rag_agent import ask


# 1. Build a small list of Golden objects by hand (input + expected_output)
#    covering both the single-turn agent and the RAG agent


# 2. Create an EvaluationDataset from your goldens


# 3. Loop over dataset.goldens, run each golden's input through the
#    matching agent, and turn each result into an LLMTestCase
#    (append each one to dataset.test_cases)


# 4. Call evaluate(dataset.test_cases, metrics=[...]) with metrics relevant
#    to each agent type


# 5. (Stretch) Use deepeval.synthesizer.Synthesizer to generate additional
#    goldens from data/knowledge_base/*.txt instead of writing them all by
#    hand -- then re-run evaluate() on the expanded dataset
