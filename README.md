# DeepEval Capstone: Testing a Single-Turn Agent, a Chatbot, and a RAG Agent

Capstone project for a DeepEval course. Three tiny "Trailhead Travel" agents
are provided as fixtures to test — the actual testing code is left for you
to write, guided by comments describing each step (DataCamp-exercise style).

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

copy .env.example .env         # then paste your real OPENAI_API_KEY into .env
```

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

Some metrics (e.g. `GEval`, `AnswerRelevancyMetric`) use an LLM as judge and
call the OpenAI API — expect these tests to cost a small amount and take a
few seconds each.

## Publishing to GitHub

Once you've filled in the tests, just ask and a git repo can be initialized
and pushed to GitHub for you (you'll need the `gh` CLI authenticated).
