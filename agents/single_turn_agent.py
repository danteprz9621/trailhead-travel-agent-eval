"""A minimal single-turn agent: one question in, one answer out, no memory.

Split into two @observe-traced sub-steps (building the prompt, calling the
LLM) plus an outer traced step for the whole agent. There's nothing to
configure to make @observe "turn on" -- the decorators pass calls straight
through in normal use, and only matter when a test drives the function
through a traced evaluation (see tests/test_tracing.py).
"""

import os

from deepeval.tracing import observe, update_current_span, update_current_trace
from google import genai
from google.genai import types

client = genai.Client()

SYSTEM_PROMPT = (
    "You are a concise assistant for 'Trailhead Travel', a fictional travel "
    "booking company. Answer travel-related questions (flights, hotels, "
    "baggage, visas) in 2-3 sentences. If a question is unrelated to travel, "
    "politely say it's outside what you can help with."
)


@observe(type="tool", name="build_prompt")
def build_messages(question: str) -> list[dict]:
    """Assemble the system + user messages sent to the LLM."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


@observe(type="llm", name="call_llm")
def call_llm(messages: list[dict], model: str = "gemini-2.5-flash") -> str:
    """Send the prepared messages to the Gemini LLM and return its reply."""
    
    # 1. Extract the system instruction if it exists in your list
    system_instruction = None
    chat_contents = []
    
    for msg in messages:
        if msg.get("role") == "system":
            system_instruction = msg.get("content")
        else:
            # Format user/model turns for Gemini
            chat_contents.append(
                types.Content(
                    role=msg.get("role"),
                    parts=[types.Part.from_text(text=msg.get("content"))]
                )
            )

    # 2. Build the configuration payload
    config = types.GenerateContentConfig(
        system_instruction=system_instruction
    )
    
    # 3. Call Google's generate_content API
    response = client.models.generate_content(
        model=model,
        contents=chat_contents,
        config=config
    )
    
    return response.text

@observe(type="agent", name="single_turn_agent")
def answer_query(question: str, model: str = "gemini-2.5-flash") -> str:
    """Send a single question to the LLM and return its one-shot answer."""
    messages = build_messages(question)
    answer = call_llm(messages, model)
    update_current_trace(input=question, output=answer)
    return answer


if __name__ == "__main__":
    print(answer_query("What's the checked baggage limit for economy flights?"))
