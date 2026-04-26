# pydantic-ai-cloudflare

PydanticAI integration for Cloudflare's AI stack — Workers AI, Browser Run, Vectorize, D1, and AI Gateway.

[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-cloudflare)](https://pypi.org/project/pydantic-ai-cloudflare/)
[![Python](https://img.shields.io/pypi/pyversions/pydantic-ai-cloudflare)](https://pypi.org/project/pydantic-ai-cloudflare/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Why This Exists

If you're building AI agents in Python, you probably use PydanticAI. If you use Cloudflare, you have access to free LLM inference, serverless web browsing, vector search, SQL storage, and request logging — but there's no PydanticAI integration for any of it.

This package connects them:

- **Free LLM inference** — Llama 3.3, Qwen 3, Kimi K2.6, Gemma 4, DeepSeek R1 — no OpenAI key needed
- **Structured output that works** — handles Workers AI quirks (dict responses, markdown fencing, truncation) automatically
- **Web browsing** — headless Chrome on the edge via Browser Run, no local browser
- **RAG** — Vectorize + Workers AI embeddings, no Pinecone
- **Conversation persistence** — D1 serverless SQLite, 5 GB free
- **Zero-config observability** — every LLM call logged via AI Gateway automatically
- **Model catalog** — `list_models()`, `recommend_model()` for discovery

Everything works on Cloudflare's free tier.

## Install

```bash
pip install pydantic-ai-cloudflare
```

## Quick Start

### One-liner agent

```python
from pydantic_ai_cloudflare import cloudflare_agent

agent = cloudflare_agent()
result = agent.run_sync("What is Cloudflare?")
print(result.output)
```

### Structured output

```python
from pydantic import BaseModel
from pydantic_ai_cloudflare import cloudflare_agent

class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    known_for: list[str]

agent = cloudflare_agent(output_type=CityInfo)
result = agent.run_sync("Tell me about Tokyo")
city = result.output  # CityInfo, not a string

print(city.name)        # "Tokyo"
print(city.population)  # 13960000
print(city.known_for)   # ["Shibuya Crossing", "Tsukiji Market", ...]
```

### Web research agent

```python
from pydantic_ai_cloudflare import cloudflare_agent

agent = cloudflare_agent(web=True)
result = agent.run_sync("What's on the Cloudflare pricing page?")
```

### With RAG

```python
agent = cloudflare_agent(web=True, rag="my-knowledge-base")
```

Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` as environment variables. That's all the config you need.

## What's Included

| Component | What it does | Cloudflare Service |
|-----------|-------------|-------------------|
| [`cloudflare_agent()`](#quick-start) | One-liner agent factory | All |
| [`cloudflare_model()`](#workers-ai-provider) | LLM inference with structured output | [Workers AI](https://developers.cloudflare.com/workers-ai/) |
| [`BrowserRunToolset`](#web-browsing-with-browser-run) | Browse, scrape, extract, crawl the web | [Browser Run](https://developers.cloudflare.com/browser-run/) |
| [`CloudflareEmbeddingModel`](#embeddings) | Text embeddings for RAG | [Workers AI Embeddings](https://developers.cloudflare.com/workers-ai/models/#text-embeddings) |
| [`VectorizeToolset`](#rag-with-vectorize) | Semantic search + knowledge storage | [Vectorize](https://developers.cloudflare.com/vectorize/) |
| [`D1MessageHistory`](#conversation-persistence-with-d1) | Conversation history across sessions | [D1](https://developers.cloudflare.com/d1/) |
| [`GatewayObservability`](#observability) | Logs, cost tracking, analytics, feedback | [AI Gateway](https://developers.cloudflare.com/ai-gateway/) |
| [`list_models()`](#model-discovery) | Browse Workers AI model catalog | — |
| [`recommend_model()`](#model-discovery) | Get the right model for your task | — |
| [`schema_stats()`](#schema-utilities) | Check schema complexity before running | — |
| [`simplify_schema()`](#schema-utilities) | Reduce schema tokens for better reliability | — |

---

## Setting Up Cloudflare

### 1. Get your Account ID

Go to [dash.cloudflare.com](https://dash.cloudflare.com). Account ID is on the right sidebar.

### 2. Create an API Token

[API Tokens](https://dash.cloudflare.com/profile/api-tokens) → **Create Token** → **Custom token**:

| Permission | Scope | Needed for |
|-----------|-------|-----------|
| **Workers AI** → Read | Account | CloudflareProvider, CloudflareEmbeddingModel |
| **Browser Rendering** → Edit | Account | BrowserRunToolset |
| **Vectorize** → Edit | Account | VectorizeToolset |
| **D1** → Edit | Account | D1MessageHistory |
| **AI Gateway** → Read | Account | GatewayObservability |

Start with just **Workers AI → Read** and **Browser Rendering → Edit**.

### 3. Set environment variables

```bash
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_API_TOKEN="your-api-token"
```

---

## Workers AI Provider

Uses the OpenAI-compatible API so any Workers AI model works with PydanticAI's full feature set — tool calling, streaming, structured output.

```python
from pydantic_ai import Agent
from pydantic_ai_cloudflare import cloudflare_model, CloudflareProvider

# Default model (Llama 3.3 70B)
agent = Agent(cloudflare_model())

# Specific model
agent = Agent(cloudflare_model("@cf/qwen/qwen3-30b-a3b"))

# With AI Gateway metadata for tracing
agent = Agent(cloudflare_model(
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    gateway_id="production",
    gateway_metadata={"team": "ml", "env": "staging"},
))

# Without AI Gateway (direct to Workers AI)
agent = Agent(cloudflare_model(gateway_id=None))
```

AI Gateway is on by default — every LLM call gets logged, cost-tracked, and shows up in your [dashboard](https://dash.cloudflare.com).

---

## Web Browsing with Browser Run

Give your agent tools to interact with any website. No Selenium, no local browser.

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_cloudflare import cloudflare_model, BrowserRunToolset

class PricingPlan(BaseModel):
    name: str
    price: str
    features: list[str]

class PricingPage(BaseModel):
    company: str
    plans: list[PricingPlan]
    has_free_tier: bool

agent = Agent(
    cloudflare_model(),
    output_type=PricingPage,
    toolsets=[BrowserRunToolset(tools=["browse", "extract"])],
)

result = agent.run_sync("Analyze pricing from https://www.cloudflare.com/plans/")
for plan in result.output.plans:
    print(f"{plan.name}: {plan.price}")
```

### Tools available

| Tool | Description | Endpoint |
|------|------------|---------|
| `browse` | Fetch page as clean markdown | [`/markdown`](https://developers.cloudflare.com/browser-run/quick-actions/markdown-endpoint/) |
| `extract` | AI-powered structured data extraction | [`/json`](https://developers.cloudflare.com/browser-run/quick-actions/json-endpoint/) |
| `crawl` | Crawl entire sites (async) | [`/crawl`](https://developers.cloudflare.com/browser-run/quick-actions/crawl-endpoint/) |
| `scrape` | CSS selector extraction | [`/scrape`](https://developers.cloudflare.com/browser-run/quick-actions/scrape-endpoint/) |
| `discover_links` | Find all links on a page | [`/links`](https://developers.cloudflare.com/browser-run/quick-actions/links-endpoint/) |
| `screenshot` | Capture screenshot (PNG) | [`/screenshot`](https://developers.cloudflare.com/browser-run/quick-actions/screenshot-endpoint/) |

---

## RAG with Vectorize

Semantic search over a knowledge base using Cloudflare Vectorize and Workers AI embeddings.

```bash
# Create a vector index (one-time setup)
npx wrangler vectorize create my-docs --dimensions 768 --metric cosine
```

```python
from pydantic_ai import Agent
from pydantic_ai_cloudflare import cloudflare_model, BrowserRunToolset, VectorizeToolset

agent = Agent(
    cloudflare_model(),
    toolsets=[
        BrowserRunToolset(tools=["browse"]),
        VectorizeToolset(index_name="my-docs"),
    ],
    system_prompt=(
        "Use browse to read web pages. Use store_knowledge to save findings. "
        "Use search_knowledge to find previously stored information."
    ),
)
```

Full pipeline on Cloudflare: `Browser Run → Workers AI embeddings → Vectorize → Workers AI`

---

## Embeddings

Use Workers AI embedding models directly with PydanticAI's Embedder system.

```python
from pydantic_ai_cloudflare import CloudflareEmbeddingModel

model = CloudflareEmbeddingModel()  # defaults to bge-base-en-v1.5, 768 dims
result = await model.embed("What is Cloudflare?", input_type="query")
print(len(result.embeddings[0]))  # 768
```

---

## Conversation Persistence with D1

```bash
# Create a D1 database (one-time setup)
npx wrangler d1 create my-chat-db
```

```python
from pydantic_ai import Agent
from pydantic_ai_cloudflare import cloudflare_model, D1MessageHistory

agent = Agent(
    cloudflare_model(),
)
history = D1MessageHistory(database_id="your-d1-database-id")

# Load previous conversation
messages = await history.get_messages("session-123")

# Run agent with context
result = await agent.run("Follow up question", message_history=messages)

# Save for next time
await history.save_messages("session-123", result.all_messages())
```

---

## Observability

Every LLM call through `CloudflareProvider` is logged via AI Gateway automatically. Query those logs programmatically:

```python
from pydantic_ai_cloudflare import GatewayObservability

obs = GatewayObservability()

logs = await obs.get_logs(limit=10)
for log in logs:
    print(f"{log['model']}: {log['tokens_in']}+{log['tokens_out']} tokens")

# Add feedback
await obs.add_feedback(logs[0]["id"], score=95, feedback=1)
```

---

## Code Mode (with Monty)

Works with [PydanticAI Code Mode](https://ai.pydantic.dev/capabilities/code-mode/). The LLM writes Python that calls your tools in parallel — Monty executes it safely.

```bash
pip install 'pydantic-ai-harness[code-mode]'
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from pydantic_ai_cloudflare import cloudflare_model, BrowserRunToolset

agent = Agent(
    cloudflare_model(),
    capabilities=[CodeMode()],
    toolsets=[BrowserRunToolset()],
)

result = agent.run_sync(
    "Compare pricing on cloudflare.com/plans and aws.amazon.com/lambda/pricing"
)
```

---

## Notebooks

Step-by-step walkthroughs in the [`notebooks/`](notebooks/) directory:

| Notebook | What you'll build |
|----------|------------------|
| [01_getting_started](notebooks/01_getting_started.ipynb) | Set up credentials, first agent, structured output |
| [02_web_research](notebooks/02_web_research.ipynb) | Browse websites, extract structured data |
| [03_rag_pipeline](notebooks/03_rag_pipeline.ipynb) | Crawl → embed → store → query (full RAG) |
| [04_persistent_chat](notebooks/04_persistent_chat.ipynb) | Multi-session conversations with D1 |

---

## Model Discovery

```python
from pydantic_ai_cloudflare import list_models, recommend_model

# See what's available
for m in list_models():
    print(f"{m['name']}: {m['context']} context, {m['speed']}")

# Filter by capability
reasoning_models = list_models(capability="reasoning")
vision_models = list_models(capability="vision")

# Get a recommendation
model = recommend_model(task="structured_output", schema_size="large")
# → "@cf/moonshotai/kimi-k2.6" (256K context, best for big schemas)
```

## Schema Utilities

For complex Pydantic schemas (18+ nested models, 9K+ chars), Workers AI models can struggle. These utilities help:

```python
from pydantic_ai_cloudflare import schema_stats, simplify_schema

# Check if your schema will work
stats = schema_stats(MyComplexModel)
print(stats)
# {'total_chars': 9066, 'simplified_chars': 3200, 'reduction': '65%',
#  'field_count': 26, 'nested_model_count': 9,
#  'recommendation': 'Large -- may need retries...'}

# Reduce schema size for better reliability
schema = MyComplexModel.model_json_schema()
simple = simplify_schema(schema)  # strips descriptions, defaults, titles
# 9066 chars → 3200 chars (65% reduction)
```

## Architecture

```
Your Python code (runs anywhere)
  │
  ├─ CloudflareProvider ──→ Workers AI ──→ AI Gateway (auto-logging)
  ├─ BrowserRunToolset ───→ Browser Run (headless Chrome on edge)
  ├─ CloudflareEmbeddingModel → Workers AI Embeddings
  ├─ VectorizeToolset ────→ Vectorize (vector database)
  ├─ D1MessageHistory ───→ D1 (serverless SQLite)
  ├─ GatewayObservability → AI Gateway REST API
  └─ CodeMode (Monty) ───→ runs in-process (<1μs startup)
```

## Roadmap

- [x] **v0.1.0** — Provider, Browser Run, Embeddings, Vectorize, D1, Gateway
- [ ] **v0.2.0** — VCR cassette integration tests, AI Search (AutoRAG) support
- [ ] **v0.3.0** — Upstream CloudflareProvider to `pydantic/pydantic-ai`
- [ ] **v1.0.0** — Stable API, full docs site

## License

MIT
