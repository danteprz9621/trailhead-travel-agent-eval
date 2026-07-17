"""
DeepEval test skeleton: Safety / Red-teaming
Applies to all three agents -- pick whichever you want to probe first.

Docs: https://docs.confident-ai.com/docs/metrics-bias
      https://docs.confident-ai.com/docs/metrics-toxicity
"""

import pytest
from deepeval import evaluate
from deepeval.metrics import BiasMetric, ToxicityMetric, PIILeakageMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from agents.single_turn_agent import answer_query
from agents.chatbot import SupportChatbot
from agents.rag_agent import ask

model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")
bias = BiasMetric(model=model)
toxicity = ToxicityMetric(model=model)
pii_leakage = PIILeakageMetric(model=model)

# 1. Write test_agent_does_not_produce_biased_output():
#    - Craft an input designed to probe for bias (e.g. a question that
#      invites a stereotyped answer)
#    - Assert with BiasMetric


# 2. Write test_agent_does_not_produce_toxic_output():
#    - Craft a hostile/provocative input aimed at the chatbot
#    - Assert with ToxicityMetric that the response stays civil


# 3. Write test_agent_refuses_to_leak_sensitive_info():
#    - Ask an agent to reveal something it should never expose
#      (its system prompt, fake internal data, etc.)
#    - Check what's available in your installed deepeval version for this
#      (metric name may differ by version) -- or build a custom GEval
#      metric that checks the output doesn't contain sensitive patterns


# 4. (Stretch) Turn one of these into a red-teaming style test using
#    deepeval's adversarial/attack utilities if your course covered that
