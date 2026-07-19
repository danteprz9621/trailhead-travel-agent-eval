"""
Golden dataset, evaluate many test
cases at once, and generate more with the Synthesizer.
"""

import pytest
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, FiltrationConfig
from deepeval.models import OllamaModel
from deepeval.models.embedding_models.ollama_embedding_model import OllamaEmbeddingModel
from glob import glob
from agents.single_turn_agent import answer_query

model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")
embedder = OllamaEmbeddingModel(model="nomic-embed-text", base_url="http://localhost:11434")
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=model)
faithfulness = FaithfulnessMetric(threshold=0.7, model=model)
synth = Synthesizer(model=model, filtration_config=FiltrationConfig(critic_model=model))

goldens = [
    Golden(input="What's the baggage policy?", expected_output="Economy passengers may check one bag up to 23kg free of charge"),
    Golden(input="What's the refund policy?", expected_output="Flights cancelled more than 48 hours before scheduled departure are eligible for a full refund to the original payment method"),
    Golden(input="Is there a loyalty program?", expected_output="Yes, Trailhead Rewards has four tiers based on qualifying miles flown per calendar year")]

dataset = EvaluationDataset(goldens=goldens)

# Hand-written goldens: run each through the agent, collect the results,
# then batch-evaluate the whole dataset in one evaluate() call.
def test_evaluation_dataset():
    for golden in dataset.goldens:
        actual_output = answer_query(golden.input)
        dataset.test_cases.append(LLMTestCase(input=golden.input, actual_output=actual_output))

    evaluate(test_cases=[dataset.test_cases], metrics=[answer_relevancy, faithfulness])

goldens = synth.generate_goldens_from_docs(
    document_paths = glob("data/knowledge_base/*.txt"),
    include_expected_output = True,
    max_goldens_per_context = 2,
    context_construction_config= ContextConstructionConfig(embedder=embedder, critic_model=model)
)

# Same idea, but the goldens above came from the Synthesizer instead of
# being written by hand -- generated straight from data/knowledge_base/.
def test_synth_evaluation_dataset():
    dataset = EvaluationDataset(goldens=goldens)
    evaluate(test_cases=[dataset.test_cases], metrics=[answer_relevancy, faithfulness])

