# DeepEval Capstone: Testing a Single-Turn Agent, a Chatbot, and a RAG Agent

Capstone project for a DeepEval course. Three tiny "Trailhead Travel" agents
are provided as fixtures to test — the actual testing code is left for you
to write, guided by comments describing each step (DataCamp-exercise style).

Runs fully local and free where set up: `single_turn_agent.py` and its test
suite use local models served by [Ollama](https://ollama.com) for both the
agent's own responses and the DeepEval judge — no API keys, no rate limits,
no cost. See [Running fully local with Ollama](#running-fully-local-with-ollama)
below.

## Project structure

```
deepeval-capstone/
├── agents/
│   ├── single_turn_agent.py   # one question in, one answer out; @observe-traced
│   ├── chatbot.py             # stateful multi-turn chatbot (SupportChatbot)
│   └── rag_agent.py           # TF-IDF retrieval over data/knowledge_base/ + LLM answer
├── data/
│   └── knowledge_base/        # 7 policy docs the RAG agent retrieves from
├── tests/
│   ├── test_single_turn_agent.py   # core metrics (AnswerRelevancy, GEval)
│   ├── test_chatbot.py             # conversational metrics (KnowledgeRetention, etc.)
│   ├── test_rag_agent.py           # RAG metrics (Faithfulness, ContextualPrecision/Recall...)
│   ├── test_safety.py              # red-teaming/safety (Bias, Toxicity)
│   ├── test_dataset_eval.py        # EvaluationDataset + Synthesizer + evaluate()
│   └── test_tracing.py             # @observe component-level tracing/evals
├── requirements.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

## Setup

```bash
cd deepeval-capstone
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`chatbot.py` and `rag_agent.py` call Gemini directly, so they need a
`GOOGLE_API_KEY` set as a real environment variable (see `.env.example` for
the name — the agent code reads `os.getenv(...)` directly, it doesn't load
`.env` files itself). `single_turn_agent.py` needs no API key at all; see
below.

### The knowledge base

`data/knowledge_base/` has 7 policy docs: `refund_policy`, `baggage_policy`,
`visa_requirements`, `booking_changes`, `travel_insurance`,
`loyalty_program`, and `flight_delays_cancellations`. Each is a few dense
paragraphs with specific numbers and edge cases, and several deliberately
cross-reference each other (e.g. travel insurance vs. refund policy, loyalty
tier vs. baggage allowance). `rag_agent.py` retrieves whole files (not
paragraph chunks) via TF-IDF, `top_k=2`.

That combination — more docs than `top_k` returns, and overlapping topics —
means retrieval is no longer "return almost everything": some questions
retrieve two genuinely relevant docs, others retrieve one relevant doc plus
one plausible-but-wrong one. That's what makes `ContextualPrecisionMetric` /
`ContextualRecallMetric` in `test_rag_agent.py` worth writing — there's
now something for them to actually penalize.

Try each agent standalone before you start testing it:

```bash
python agents/single_turn_agent.py
python agents/chatbot.py
python agents/rag_agent.py
```

## Running fully local with Ollama

`single_turn_agent.py` and `tests/test_single_turn_agent.py` run entirely on
local models via [Ollama](https://ollama.com) — no OpenAI/Gemini API key
needed for that pair, and no rate limits or per-request cost. This came out
of hitting two real problems while writing the tests:

1. **Free-tier rate limits.** Cloud judge models (and the agent's own calls)
   burned through free-tier quotas fast — Gemini's free tier caps
   `gemini-2.5-flash` at just 20 requests *per day*, which a handful of
   pytest runs exhausts immediately.
2. **Self-evaluation bias.** Using the same model as both the agent
   generating an answer and the judge scoring it risks the judge being blind
   to that model's own failure patterns.

The fix: two different local models, one per role, both served by Ollama on
`http://localhost:11434`:

- **Agent** (`agents/single_turn_agent.py`) generates answers with
  `qwen2.5-coder:7b`.
- **Judge** (`tests/test_single_turn_agent.py`) scores those answers with a
  *different* model, `llama3.1:8b`, via `deepeval`'s `OllamaModel`:

  ```python
  from deepeval.models import OllamaModel

  judge_model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")
  answer_relevancy = AnswerRelevancyMetric(model=judge_model)
  ```

To run it yourself: install [Ollama](https://ollama.com), then pull both
models and make sure the Ollama server is running (it typically runs as a
background service after install):

```bash
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
```

`chatbot.py` and `rag_agent.py` still call Gemini directly (`google-genai`)
for their generation, so they'll need a `GOOGLE_API_KEY` and are still
subject to that same daily free-tier cap — the same agent/judge Ollama split
can be applied to them if you hit it.

## How the test files work

Each file under `tests/` is a **skeleton**, not a finished test suite. Every
section is a numbered comment describing one thing to implement — imports
are already set up, but the metric instances, `LLMTestCase`/
`ConversationalTestCase` objects, and assertions are yours to write. Work
through them in this order, since each builds on DeepEval concepts from the
previous one:

1. `test_single_turn_agent.py` — single `LLMTestCase`s, `AnswerRelevancyMetric`, a custom `GEval` metric
2. `test_chatbot.py` — `ConversationalTestCase`/`Turn`, `KnowledgeRetentionMetric`, `ConversationCompletenessMetric`, `RoleAdherenceMetric`
3. `test_rag_agent.py` — `retrieval_context`, `FaithfulnessMetric`, `ContextualPrecisionMetric`/`ContextualRecallMetric`, `HallucinationMetric`
4. `test_safety.py` — `BiasMetric`, `ToxicityMetric`, and a custom metric for sensitive-data leakage
5. `test_dataset_eval.py` — `Golden`, `EvaluationDataset`, bulk `evaluate()`, and (stretch) the `Synthesizer`
6. `test_tracing.py` — component-level evals via `@observe`, scoring individual steps inside `single_turn_agent.py` (prompt-building vs. the LLM call) instead of only the end-to-end output. Do this one last — it builds on the `LLMTestCase`/`Golden`/`EvaluationDataset` concepts from steps 1 and 5.

## Running tests

```bash
pytest tests/ -v
# or, for deepeval's own CLI runner/reporting:
deepeval test run tests/
```

Every metric (e.g. `GEval`, `AnswerRelevancyMetric`) uses an LLM as judge.
`test_single_turn_agent.py` points its judge at a local Ollama model (see
above), so those runs are free — expect roughly 15-70s total depending on
whether Ollama has to swap models in/out of VRAM. For test files you write
against a cloud judge model instead, expect a small per-run cost and a few
seconds per test case.
