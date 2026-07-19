# DeepEval Practice Project: Testing a Single-Turn Agent, a Chatbot, and a RAG Agent

A practice project for testing LLM-backed agents with
[DeepEval](https://deepeval.com). Three tiny "Trailhead Travel" agents (a
single-turn Q&A agent, a multi-turn chatbot, and a RAG agent) act as fixtures,
with a DeepEval test suite covering core metrics, conversational metrics,
RAG metrics, safety/red-teaming, dataset-based evaluation with synthetic
data, and component-level tracing.

Runs fully local and free: every agent and every judge/embedder model is
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

No API key is required anywhere in this project — every agent and every
judge/embedder model runs locally via Ollama (see below). `.env.example` is
unused/vestigial at this point; it's kept as a placeholder in case you ever
point something at a cloud provider instead.

### The knowledge base

`data/knowledge_base/` has 7 policy docs: `refund_policy`, `baggage_policy`,
`visa_requirements`, `booking_changes`, `travel_insurance`,
`loyalty_program`, and `flight_delays_cancellations`. Each is a few dense
paragraphs with specific numbers and edge cases, and several deliberately
cross-reference each other (e.g. travel insurance vs. refund policy, loyalty
tier vs. baggage allowance). `rag_agent.py` retrieves whole files (not
paragraph chunks) via TF-IDF, `top_k=2`.

That combination — more docs than `top_k` returns, and overlapping topics —
means retrieval doesn't just "return almost everything": some questions
retrieve two genuinely relevant docs, others retrieve one relevant doc plus
one plausible-but-wrong one. That gives `ContextualPrecisionMetric` /
`ContextualRecallMetric` in `test_rag_agent.py` something real to penalize.

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
cost. Two problems make this worth doing over a cloud provider:

1. **Free-tier rate limits.** Cloud judge models (and the agents' own calls)
   burn through free-tier quotas fast — e.g. Gemini's free tier caps
   `gemini-2.5-flash` at just 20 requests *per day* (and separately, 5 per
   *minute*), which a handful of pytest runs can exhaust immediately.
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

`test_dataset_eval.py` adds a third role: its `Synthesizer` (which generates
synthetic goldens from `data/knowledge_base/`) needs its own **embedding**
model, separate from the generation/judge models — DeepEval defaults that to
OpenAI too if you don't override it. `nomic-embed-text` fills that role: it's
a small model purpose-built for embeddings (general chat models like
`llama3.1:8b` don't support Ollama's embeddings endpoint at all).

To run it yourself: install [Ollama](https://ollama.com), then pull all
three models and make sure the Ollama server is running (it typically runs
as a background service after install):

```bash
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Since `qwen2.5-coder:7b` and `llama3.1:8b` are roughly the same size (~5GB
each) and a typical 8GB GPU can't hold both at once, expect Ollama to swap
models in/out of VRAM between an agent call and a judge call — a few seconds
of overhead per swap, not a problem for iterating on tests but worth knowing
about if a run feels slower than expected.

`test_dataset_eval.py`'s `Synthesizer` also pulls in a few extra Python
packages beyond what the rest of the project uses — `chromadb` (vector store
for chunking source docs) and `langchain` / `langchain_community` /
`langchain_text_splitters` (document loading). These are all in
`requirements.txt`.

## Running tests

```bash
pytest tests/ -v
# or, for deepeval's own CLI runner/reporting:
deepeval test run tests/
```

Every metric (e.g. `GEval`, `AnswerRelevancyMetric`) uses an LLM as judge.
All six test files point their judges at a local Ollama model (see above),
so runs are free — expect anywhere from ~4s (`.deepeval-cache.json` happens
to have a matching cached result) to over a minute (fresh evaluation with
VRAM swapping, or `test_dataset_eval.py`'s synthetic golden generation).
