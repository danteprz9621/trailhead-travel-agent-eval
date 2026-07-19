"""A minimal single-turn agent: one question in, one answer out, no memory.

Split into two @observe-traced sub-steps (building the prompt, calling the
LLM) plus an outer traced step for the whole agent. There's nothing to
configure to make @observe "turn on" -- the decorators pass calls straight
through in normal use, and only matter when a test drives the function
through a traced evaluation (see tests/test_tracing.py).
"""

import os

import ollama
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.metrics import AnswerRelevancyMetric, PromptAlignmentMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

SYSTEM_PROMPT = (
    "You are a concise assistant for 'Trailhead Travel', a fictional travel "
    "booking company. Answer travel-related questions (flights, hotels, "
    "baggage, visas) in 2-3 sentences. If a question is unrelated to travel, "
    "politely say it's outside what you can help with."
)

judge_model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "Include the Trailhead Travel system persona as the system message.",
        "Include the user's exact question, unmodified, as the user message.",
    ],
    model=judge_model,
)
answer_relevancy = AnswerRelevancyMetric(model=judge_model)


@observe(type="tool", name="build_prompt", metrics=[prompt_alignment])
def build_messages(question: str) -> list[dict]:
    """Assemble the system + user messages sent to the LLM."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


@observe(type="llm", name="call_llm", metrics=[answer_relevancy])
def call_llm(messages: list[dict], model: str = "qwen2.5-coder:7b") -> str:
    """Send the prepared messages to the local Ollama LLM and return its reply."""
    response = ollama.chat(model=model, messages=messages)
    update_current_span(test_case=LLMTestCase(
        input=messages[1]["content"],
        actual_output=response["message"]["content"]))
    return response["message"]["content"]


@observe(type="agent", name="single_turn_agent")
def answer_query(question: str, model: str = "qwen2.5-coder:7b") -> str:
    """Send a single question to the LLM and return its one-shot answer."""
    messages = build_messages(question)
    answer = call_llm(messages, model)
    update_current_trace(input=question, output=answer)
    return answer


if __name__ == "__main__":
    print(answer_query("What's the checked baggage limit for economy flights?"))
