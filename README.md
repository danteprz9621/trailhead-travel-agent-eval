# DeepEval Capstone: Testing a Single-Turn Agent, a Chatbot, and a RAG Agent

Capstone project for a DeepEval course. Three tiny "Trailhead Travel" agents
are provided as fixtures to test. `test_single_turn_agent.py`, `test_chatbot.py`,
`test_rag_agent.py`, and `test_safety.py` are written; `test_dataset_eval.py`
and `test_tracing.py` are still skeletons — see
[How the test files work](#how-the-test-files-work) below.

Runs fully local and free: all three agents and their judge models are
served by [Ollama](https://ollama.com) — no API keys, no rate limits, no
cost. See [Running fully local with Ollama](#running-fully-local-with-ollama)
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

No API key is required for any agent right now — all three run on local
Ollama models (see below). `.env.example` is currently unused/vestigial;
it's there as a placeholder if a future test file (e.g. `test_dataset_eval.py`)
ends up needing a cloud provider key.

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

All three agents (`single_turn_agent.py`, `chatbot.py`, `rag_agent.py`) and
every judge model in the test files run entirely on local models via
[Ollama](https://ollama.com) — no API keys, no rate limits, no per-request
cost. This came out of hitting two real problems while writing the tests
against Gemini originally:

1. **Free-tier rate limits.** Cloud judge models (and the agents' own calls)
   burned through free-tier quotas fast — Gemini's free tier caps
   `gemini-2.5-flash` at just 20 requests *per day* (and separately, 5 per
   *minute*), which a handful of pytest runs exhausts immediately.
2. **Self-evaluation bias.** Using the same model as both the agent
   generating an answer and the judge scoring it risks the judge being blind
   to that model's own failure patterns.

The fix: two different local models, one per role, both served by Ollama on
`http://localhost:11434`:

- **Agents** (`single_turn_agent.py`, `chatbot.py`, `rag_agent.py`) generate
  with `qwen2.5-coder:7b`.
- **Judges** (in every test file) score those answers with a *different*
  model, `llama3.1:8b`, via `deepeval`'s `OllamaModel`:

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

Since both models are roughly the same size (~5GB each) and a typical 8GB
GPU can't hold both at once, expect Ollama to swap models in/out of VRAM
between an agent call and a judge call — a few seconds of overhead per
swap, not a problem for iterating on tests but worth knowing about if a run
feels slower than expected.

## How the test files work

Originally, every file under `tests/` was a **skeleton** — numbered comments
describing one thing to implement, no actual metric/assertion code. That's
still true for the last two; the first four are fully written:

1. ✅ `test_single_turn_agent.py` — single `LLMTestCase`s, `AnswerRelevancyMetric`, a custom `GEval` metric
2. ✅ `test_chatbot.py` — `ConversationalTestCase`/`Turn`, `KnowledgeRetentionMetric`, `ConversationCompletenessMetric`, `RoleAdherenceMetric`, a conversational `GEval`
3. ✅ `test_rag_agent.py` — `retrieval_context` vs. `context`, `FaithfulnessMetric`, `ContextualPrecisionMetric`/`ContextualRecallMetric`, `HallucinationMetric`
4. ✅ `test_safety.py` — `BiasMetric`, `ToxicityMetric`, `PIILeakageMetric`, and a custom `GEval` for prompt-injection/system-prompt leakage (not PII, since no personal data is involved there)
5. ⬜ `test_dataset_eval.py` — `Golden`, `EvaluationDataset`, bulk `evaluate()`, and (stretch) the `Synthesizer`
6. ⬜ `test_tracing.py` — component-level evals via `@observe`, scoring individual steps inside `single_turn_agent.py` (prompt-building vs. the LLM call) instead of only the end-to-end output. Builds on the `LLMTestCase`/`Golden`/`EvaluationDataset` concepts from steps 1 and 5.

## Running tests

```bash
pytest tests/ -v
# or, for deepeval's own CLI runner/reporting:
deepeval test run tests/
```

Every metric (e.g. `GEval`, `AnswerRelevancyMetric`) uses an LLM as judge.
All four written test files point their judges at a local Ollama model (see
above), so runs are free — expect anywhere from ~4s (`.deepeval-cache.json`
happens to have a matching cached result) to ~50s (fresh evaluation with
VRAM swapping) per file. For any test file you write against a cloud judge
model instead, expect a small per-run cost and a few seconds per test case.
