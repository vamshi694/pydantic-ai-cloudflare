# pydantic-ai-cloudflare

**The PydanticAI SDK for Cloudflare's AI stack.**

Build Python AI agents with type-safe structured output, web browsing, RAG, conversation persistence, and zero-config observability — entirely on Cloudflare's free tier.

[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-cloudflare)](https://pypi.org/project/pydantic-ai-cloudflare/)
[![Python](https://img.shields.io/pypi/pyversions/pydantic-ai-cloudflare)](https://pypi.org/project/pydantic-ai-cloudflare/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-101%20passing-brightgreen)](tests/)

```bash
pip install pydantic-ai-cloudflare
```

```python
from pydantic_ai_cloudflare import cloudflare_agent

agent = cloudflare_agent()
result = agent.run_sync("What is Cloudflare?")
```

---

## What Cloudflare Already Has

Cloudflare provides a complete AI infrastructure stack — **all with free tiers**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLOUDFLARE AI INFRASTRUCTURE                     │
├─────────────────┬───────────────────┬───────────────────────────────┤
│                 │                   │                               │
│  ┌───────────┐  │  ┌─────────────┐  │  ┌──────────────────────────┐ │
│  │Workers AI │  │  │ Browser Run │  │  │      AI Gateway          │ │
│  │           │  │  │             │  │  │                          │ │
│  │ 20+ LLMs │  │  │  Headless   │  │  │  Logging · Analytics     │ │
│  │ Embedding │  │  │  Chrome on  │  │  │  Cost tracking · Cache   │ │
│  │ Free tier │  │  │  the edge   │  │  │  Rate limiting           │ │
│  └───────────┘  │  └─────────────┘  │  └──────────────────────────┘ │
│                 │                   │                               │
│  ┌───────────┐  │  ┌─────────────┐  │  ┌──────────────────────────┐ │
│  │ Vectorize │  │  │     D1      │  │  │        R2                │ │
│  │           │  │  │             │  │  │                          │ │
│  │  Vector   │  │  │ Serverless  │  │  │  Object storage          │ │
│  │  database │  │  │   SQLite    │  │  │  Zero egress fees        │ │
│  │  for RAG  │  │  │   5GB free  │  │  │  10GB free               │ │
│  └───────────┘  │  └─────────────┘  │  └──────────────────────────┘ │
│                 │                   │                               │
└─────────────────┴───────────────────┴───────────────────────────────┘
```

**The problem**: There's no Python SDK that connects PydanticAI to any of this. Until now.

## What This Library Does

```
┌──────────────────────────────────────────────────────────────────────┐
│                     pydantic-ai-cloudflare                           │
│                                                                      │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │              │  │                  │  │                       │  │
│  │ cloudflare_  │  │ BrowserRun       │  │ VectorizeToolset      │  │
│  │ agent()      │  │ Toolset          │  │                       │  │
│  │              │  │                  │  │ search_knowledge()    │  │
│  │ One-liner    │  │ browse()         │  │ store_knowledge()     │  │
│  │ agent        │  │ extract()        │  │                       │  │
│  │ factory      │  │ crawl()          │  │ Workers AI embeddings │  │
│  │              │  │ scrape()         │  │ + Vectorize storage   │  │
│  └──────┬───────┘  │ discover_links() │  └───────────┬───────────┘  │
│         │          │ screenshot()     │              │              │
│         │          └────────┬─────────┘              │              │
│         │                   │                        │              │
│  ┌──────┴───────────────────┴────────────────────────┴───────────┐  │
│  │                                                               │  │
│  │  CloudflareProvider  ────────→  Workers AI  ──→  AI Gateway   │  │
│  │  (auto AI Gateway routing, response normalization,            │  │
│  │   model profiles for all Workers AI model families)           │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐  │
│  │ D1MessageHistory  │  │ GatewayObserv.    │  │ Schema Utils    │  │
│  │                   │  │                   │  │                 │  │
│  │ Conversation      │  │ get_logs()        │  │ simplify_schema │  │
│  │ persistence       │  │ get_analytics()   │  │ schema_stats()  │  │
│  │ across sessions   │  │ add_feedback()    │  │ extract_json()  │  │
│  └───────────────────┘  └───────────────────┘  └─────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | What it does | Cloudflare Service |
|-----------|-------------|-------------------|
| `cloudflare_agent()` | One-liner agent factory with sensible defaults | All |
| `cloudflare_model()` | LLM inference with auto response normalization | [Workers AI](https://developers.cloudflare.com/workers-ai/) |
| `BrowserRunToolset` | 6 web interaction tools for agents | [Browser Run](https://developers.cloudflare.com/browser-run/) |
| `VectorizeToolset` | RAG search + store (DIY) | [Vectorize](https://developers.cloudflare.com/vectorize/) |
| `AISearchToolset` | Managed RAG search + chat | [AI Search](https://developers.cloudflare.com/ai-search/) |
| `CloudflareEmbeddingModel` | Text embeddings | [Workers AI](https://developers.cloudflare.com/workers-ai/models/#text-embeddings) |
| `D1MessageHistory` | Conversation persistence | [D1](https://developers.cloudflare.com/d1/) |
| `GatewayObservability` | Logs, cost, analytics, feedback | [AI Gateway](https://developers.cloudflare.com/ai-gateway/) |
| `list_models()` / `recommend_model()` | Model discovery + recommendations | — |
| `cf_structured()` | Complex structured output that works on ALL models | [Workers AI](https://developers.cloudflare.com/workers-ai/) |
| `simplify_schema()` / `schema_stats()` | Schema optimization for reliability | — |

### What we handle that's hard

Workers AI has quirks that break naive integrations. This library handles them:

- **Dict content responses** — Workers AI returns `content` as a parsed dict instead of a JSON string. We normalize it.
- **Markdown code fences** — Models wrap JSON in ` ```json ... ``` `. We strip them.
- **Prose-wrapped JSON** — Models add "Here's the JSON:" before the actual JSON. We extract it.
- **Model-specific structured output** — Each model family needs a different strategy (tool calling vs json_object vs guided_json). Our profiles handle this automatically.
- **Schema simplification** — Large schemas (9K+ chars) overwhelm models. `simplify_schema()` strips descriptions and defaults (65% reduction) while keeping the structure valid.

---

## Quick Start

### 1. Set up Cloudflare credentials

```bash
# Get your Account ID from https://dash.cloudflare.com (right sidebar)
export CLOUDFLARE_ACCOUNT_ID="your-account-id"

# Create an API token at https://dash.cloudflare.com/profile/api-tokens
# Permissions: Workers AI → Read, Browser Rendering → Edit
export CLOUDFLARE_API_TOKEN="your-api-token"
```

### What each feature needs

| Feature | Token Permission | CF Resource Needed | How to Create |
|---------|-----------------|-------------------|---------------|
| `cloudflare_agent()` | Workers AI Read | None | — |
| `cf_structured()` | Workers AI Read | None | — |
| `BrowserRunToolset` | Browser Rendering Edit | None | — |
| `VectorizeToolset` | Vectorize Edit | A Vectorize index | `npx wrangler vectorize create NAME --dimensions 768 --metric cosine` |
| `AISearchToolset` | AI Search Edit + Run | An AI Search instance | Dashboard → AI → AI Search → Create |
| `CloudflareEmbeddingModel` | Workers AI Read | None | — |
| `D1MessageHistory` | D1 Edit | A D1 database | `npx wrangler d1 create NAME` |
| `GatewayObservability` | AI Gateway Read | None (auto-created) | — |

Start with just **Workers AI Read** + **Browser Rendering Edit**. Add more as you need them.

### 2. Install

```bash
pip install pydantic-ai-cloudflare
```

### 3. Use

```python
from pydantic_ai_cloudflare import cloudflare_agent

# Plain text
agent = cloudflare_agent()
result = agent.run_sync("What is Cloudflare?")
print(result.output)

# Structured output
from pydantic import BaseModel
class City(BaseModel):
    name: str
    country: str
    population: int

agent = cloudflare_agent(output_type=City)
result = agent.run_sync("Tell me about Tokyo")
print(result.output.name)        # "Tokyo"
print(result.output.population)  # 13900000

# With web browsing
agent = cloudflare_agent(web=True)
result = agent.run_sync("What's on cloudflare.com/plans?")

# With RAG
agent = cloudflare_agent(web=True, rag="my-knowledge-base")

# Specific model
agent = cloudflare_agent(model="@cf/qwen/qwen3-30b-a3b")
```

---

## Code Mode with Monty

[Monty](https://github.com/pydantic/monty) is PydanticAI's sandboxed Python interpreter. Instead of the LLM making 10 sequential tool calls (10 round-trips), it writes **one Python script** that calls your tools in parallel. Monty executes it safely in <1μs.

```
┌──────────────────────────────────────────────────────────────────┐
│                    WITHOUT Code Mode                              │
│                                                                   │
│  LLM call 1 → browse(cloudflare.com/plans)     → wait for result │
│  LLM call 2 → browse(aws.amazon.com/pricing)   → wait for result │
│  LLM call 3 → extract(cloudflare.com/plans)    → wait for result │
│  LLM call 4 → extract(aws.amazon.com/pricing)  → wait for result │
│  LLM call 5 → compare results                  → wait for result │
│  LLM call 6 → generate report                  → final answer    │
│                                                                   │
│  Total: 6 LLM round-trips, ~30 seconds                           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    WITH Code Mode (Monty)                         │
│                                                                   │
│  LLM call 1 → writes Python:                                     │
│    ┌──────────────────────────────────────────────────┐           │
│    │ cf, aws = await asyncio.gather(                  │           │
│    │     browse("cloudflare.com/plans"),              │           │
│    │     browse("aws.amazon.com/pricing"),            │           │
│    │ )                                                │           │
│    │ cf_data = await extract(cf, "pricing plans")     │           │
│    │ aws_data = await extract(aws, "pricing plans")   │           │
│    │ return compare(cf_data, aws_data)                │           │
│    └──────────────────────────────────────────────────┘           │
│  Monty executes it (<1μs) → tools run in parallel → done         │
│                                                                   │
│  Total: 1-2 LLM round-trips, ~10 seconds                         │
└──────────────────────────────────────────────────────────────────┘
```

```bash
pip install 'pydantic-ai-harness[code-mode]'
```

```python
from pydantic_ai_harness import CodeMode
from pydantic_ai_cloudflare import cloudflare_agent

agent = cloudflare_agent(
    web=True,
    capabilities=[CodeMode()],
)

result = agent.run_sync(
    "Compare pricing on cloudflare.com/plans and aws.amazon.com/lambda/pricing"
)
```

The LLM writes Python, Monty executes it in a sandbox, your tools (Browser Run, Vectorize, etc.) run on Cloudflare's edge. Best of both worlds.

---

## Model Discovery

Don't know which Workers AI model to use? Let the library recommend one:

```python
from pydantic_ai_cloudflare import list_models, recommend_model

# Browse the catalog
for m in list_models():
    print(f"{m['name']}: {m['context']} context, {m['speed']}")
# Llama 3.3 70B: 128K context, fast
# Qwen 3 30B: 128K context, fast
# Kimi K2.6: 256K context, medium
# ...

# Filter by capability
list_models(capability="reasoning")  # → Qwen 3, Kimi, DeepSeek R1, ...
list_models(capability="vision")     # → Gemma 4, Llama 3.2 Vision

# Get a recommendation
recommend_model(task="reasoning")         # → Qwen 3 30B
recommend_model(task="vision")            # → Gemma 4 26B
recommend_model(schema_size="large")      # → Kimi K2.6 (256K context)
recommend_model(speed="very_fast")        # → Llama 3.1 8B
```

---

## Web Browsing

```python
from pydantic_ai_cloudflare import cloudflare_agent

agent = cloudflare_agent(web=True)
result = agent.run_sync("Summarize the Cloudflare Workers AI docs page")
```

The agent has 6 tools:

| Tool | What it does | Use case |
|------|-------------|----------|
| `browse` | Fetch page as markdown | Read any webpage |
| `extract` | AI-powered JSON extraction | Pull structured data from a page |
| `crawl` | Crawl entire sites | Build knowledge bases |
| `scrape` | CSS selector extraction | Grab specific elements |
| `discover_links` | Find all links | Explore a site |
| `screenshot` | Capture PNG | Visual QA |

---

## RAG with Vectorize

```bash
npx wrangler vectorize create my-docs --dimensions 768 --metric cosine
```

```python
from pydantic_ai_cloudflare import cloudflare_agent

agent = cloudflare_agent(
    web=True,
    rag="my-docs",
    system_prompt="Browse pages, store findings, answer from knowledge base.",
)
```

Full pipeline: `Browser Run → Workers AI embeddings → Vectorize → Workers AI`

---

## AI Search (Managed RAG)

If you don't want to manage embeddings and Vectorize yourself, use [AI Search](https://developers.cloudflare.com/ai-search/) -- Cloudflare's fully-managed RAG. Point it at an R2 bucket or website, and it handles chunking, embedding, indexing, and search.

Create an instance in the dashboard: AI → AI Search → Create

```python
from pydantic_ai_cloudflare import cloudflare_agent, AISearchToolset

agent = cloudflare_agent(
    toolsets=[AISearchToolset(instance_name="my-docs")],
)
result = agent.run_sync("What does our documentation say about caching?")
```

The agent gets two tools: `search` (returns relevant chunks) and `ask` (returns an AI-generated answer with citations).

---

## Conversation Persistence

```bash
npx wrangler d1 create my-chat-db
```

```python
from pydantic_ai_cloudflare import cloudflare_agent, D1MessageHistory

agent = cloudflare_agent()
history = D1MessageHistory(database_id="your-d1-uuid")

messages = await history.get_messages("session-123")
result = await agent.run("Follow up question", message_history=messages)
await history.save_messages("session-123", result.all_messages())
```

---

## Observability

Every LLM call through `cloudflare_agent()` is logged via AI Gateway automatically. Query programmatically:

```python
from pydantic_ai_cloudflare import GatewayObservability

obs = GatewayObservability()
logs = await obs.get_logs(limit=10)
await obs.add_feedback(logs[0]["id"], score=95, feedback=1)
```

Or just check [dash.cloudflare.com](https://dash.cloudflare.com) → AI → AI Gateway.

---

## Schema Utilities

For complex Pydantic models, check reliability before running:

```python
from pydantic_ai_cloudflare import schema_stats, simplify_schema

stats = schema_stats(MyComplexModel)
# {'total_chars': 9066, 'simplified_chars': 3200, 'reduction': '65%',
#  'field_count': 26, 'nested_model_count': 9,
#  'recommendation': 'Large -- may need retries...'}
```

---

## Complex Structured Output — `cf_structured()`

PydanticAI's built-in structured output uses tool calling, which breaks on Workers AI for complex schemas (null arguments, malformed retries). `cf_structured()` bypasses this and calls Workers AI directly with the same approach as `langchain-cloudflare`:

```python
from pydantic_ai_cloudflare import cf_structured_sync

result = cf_structured_sync(
    "Research report on NovaPay, a payment processing startup",
    CompanyReport,  # 7 nested models, Literal types, lists
    model="@cf/qwen/qwen3-30b-a3b-fp8",
)
print(result.company.name)   # validated Pydantic object
print(result.next_steps[0])  # NextStep(action=..., priority="HIGH")
```

How it works:
1. Generates + simplifies JSON schema from your Pydantic model
2. Injects schema into system prompt with strict formatting instructions
3. Sets `response_format: json_object` to force valid JSON
4. Parses response (handles dict content, markdown fences, prose wrapping)
5. Validates against Pydantic
6. On failure: retries with error feedback (not via API messages that Workers AI rejects)

**Tested on all 6 major Workers AI models** with a 7-nested-model schema:

| Model | Complex Schema (7 nested) | Time |
|---|---|---|
| Llama 3.3 70B | Pass | 31s |
| Qwen 3 30B | Pass | 17s |
| Kimi K2.6 | Pass | 55s |
| Gemma 4 26B | Pass | 32s |
| GLM 4.7 Flash | Pass | 24s |
| DeepSeek R1 32B | Pass | 30s |

**When to use what:**
- Simple schemas (3-5 fields): `cloudflare_agent(output_type=MyModel)` works fine
- Complex schemas (4+ nested models, Literal types): use `cf_structured()`

---

## Notebooks

| Notebook | What you'll learn | Has outputs? |
|----------|------------------|:---:|
| [01_getting_started](notebooks/01_getting_started.ipynb) | First agent, structured output, model discovery | Yes |
| [02_web_research](notebooks/02_web_research.ipynb) | Browse, extract, discover links, scrape | Yes |
| [03_rag_pipeline](notebooks/03_rag_pipeline.ipynb) | Crawl → embed → store → query with Vectorize | Template |
| [04_persistent_chat](notebooks/04_persistent_chat.ipynb) | Multi-session conversations with D1 | Template |
| [05_code_mode_monty](notebooks/05_code_mode_monty.ipynb) | Parallel tool execution with Monty | Walkthrough |
| [06_complex_structured_output](notebooks/06_complex_structured_output.ipynb) | `cf_structured()` across all Workers AI models | Yes |

---

## How It Compares

| | pydantic-ai-cloudflare | langchain-cloudflare | Raw API calls |
|---|---|---|---|
| **Framework** | PydanticAI | LangChain | None |
| **Type safety** | Full Pydantic models | Loose | Manual |
| **Structured output** | Automatic (handles Workers AI quirks) | Manual method choice | DIY |
| **Response normalization** | Built-in (dict, fences, prose) | Built-in | DIY |
| **Agent factory** | `cloudflare_agent()` one-liner | No | No |
| **Model discovery** | `list_models()`, `recommend_model()` | No | No |
| **Schema optimization** | `simplify_schema()`, `schema_stats()` | No | No |
| **Web browsing** | `BrowserRunToolset` (6 tools) | Loader + Tool | httpx calls |
| **RAG** | `VectorizeToolset` (2 tools) | CloudflareVectorize | Multiple APIs |
| **Persistence** | `D1MessageHistory` | D1Saver (checkpoint only) | SQL queries |
| **Observability** | Auto via AI Gateway | None | Manual logging |
| **Code Mode** | Works with Monty | No | No |
| **Cost** | Free tier | Free tier | Free tier |

---

## Roadmap

- [x] **v0.1.0** — Provider, Browser Run, Embeddings, Vectorize, D1, Gateway, Model Catalog, Schema Utils
- [ ] **v0.2.0** — VCR cassette integration tests, AI Search (AutoRAG) support
- [ ] **v0.3.0** — Upstream CloudflareProvider to `pydantic/pydantic-ai`
- [ ] **v1.0.0** — Stable API, full docs site, PyPI release

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
