# pydantic-ai-cloudflare

PydanticAI integration for Cloudflare's AI stack.

Build AI agents with **type-safe structured output**, **web browsing**, **RAG**, **conversation persistence**, and **zero-config observability** — entirely on Cloudflare's infrastructure.

```bash
pip install pydantic-ai-cloudflare
```

## What This Gives You

| Component | What it does | Cloudflare Service |
|-----------|-------------|-------------------|
| **CloudflareProvider** | LLM inference (tool calling, streaming, structured output) | [Workers AI](https://developers.cloudflare.com/workers-ai/) |
| **BrowserRunToolset** | Browse, scrape, extract, crawl the web | [Browser Run](https://developers.cloudflare.com/browser-run/) |
| **CloudflareEmbeddingModel** | Text embeddings for RAG | [Workers AI Embeddings](https://developers.cloudflare.com/workers-ai/models/#text-embeddings) |
| **VectorizeToolset** | Semantic search + knowledge storage | [Vectorize](https://developers.cloudflare.com/vectorize/) |
| **D1MessageHistory** | Conversation persistence across sessions | [D1](https://developers.cloudflare.com/d1/) |
| **GatewayObservability** | Logs, cost tracking, analytics, feedback | [AI Gateway](https://developers.cloudflare.com/ai-gateway/) |

All components work on the **free tier**. No credit card required to start.

---

## Setting Up Cloudflare

### Step 1: Get your Account ID

Go to the [Cloudflare dashboard](https://dash.cloudflare.com). Your Account ID is on the right sidebar of the overview page. Copy it.

### Step 2: Create an API Token

Go to [API Tokens](https://dash.cloudflare.com/profile/api-tokens) → **Create Token** → **Custom token**.

Add these permissions based on what you need:

| Permission | Scope | Required for |
|-----------|-------|-------------|
| **Workers AI** → Read | Account | `CloudflareProvider`, `CloudflareEmbeddingModel` |
| **Browser Rendering** → Edit | Account | `BrowserRunToolset` |
| **Vectorize** → Edit | Account | `VectorizeToolset` |
| **D1** → Edit | Account | `D1MessageHistory` |
| **AI Gateway** → Read | Account | `GatewayObservability` |

For getting started, just add **Workers AI → Read** and **Browser Rendering → Edit**.

### Step 3: Set environment variables

```bash
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_API_TOKEN="your-api-token"
```

Or put them in a `.env` file and load with `python-dotenv`.

### Step 4 (Optional): Create Cloudflare resources

For RAG (Vectorize):
```bash
# Install wrangler if you don't have it
npm install -g wrangler

# Create a vector index
npx wrangler vectorize create my-knowledge-base --dimensions 768 --metric cosine
```

For conversation persistence (D1):
```bash
npx wrangler d1 create my-conversations
```

---

## Quick Start — Simple Chat

```python
from pydantic_ai import Agent

agent = Agent("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast")
result = agent.run_sync("What is Cloudflare?")
print(result.output)
```

That's it. Inference runs on Workers AI. Logs appear automatically in your [AI Gateway dashboard](https://dash.cloudflare.com) under the `default` gateway.

---

## Structured Output

Your LLM output is a **validated Pydantic model**, not a string you have to parse.

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class CompanyProfile(BaseModel):
    name: str
    industry: str
    founded_year: int
    key_products: list[str]

agent = Agent(
    "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    output_type=CompanyProfile,
)

result = agent.run_sync("Tell me about Cloudflare")
company = result.output  # ← CompanyProfile instance, not a string

print(company.name)          # "Cloudflare"
print(company.founded_year)  # 2009
print(company.key_products)  # ["Workers", "R2", "D1", ...]
```

Works with any Pydantic model — nested objects, lists, enums, optional fields. PydanticAI handles schema conversion, validation, and automatic retries on malformed output.

---

## Web Browsing with Browser Run

Give your agent the ability to interact with any website.

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_cloudflare import BrowserRunToolset

class PricingPlan(BaseModel):
    name: str
    price: str
    features: list[str]

class PricingAnalysis(BaseModel):
    company: str
    plans: list[PricingPlan]
    has_free_tier: bool

agent = Agent(
    "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    output_type=PricingAnalysis,
    toolsets=[BrowserRunToolset(tools=["browse", "extract"])],
)

result = agent.run_sync("Analyze pricing from cloudflare.com/plans")
for plan in result.output.plans:
    print(f"{plan.name}: {plan.price}")
```

### Available tools

| Tool | Description | Browser Run Endpoint |
|------|------------|---------------------|
| `browse` | Fetch any page as clean markdown | [`/markdown`](https://developers.cloudflare.com/browser-run/quick-actions/markdown-endpoint/) |
| `extract` | AI-powered structured data extraction | [`/json`](https://developers.cloudflare.com/browser-run/quick-actions/json-endpoint/) |
| `crawl` | Crawl an entire site (async polling) | [`/crawl`](https://developers.cloudflare.com/browser-run/quick-actions/crawl-endpoint/) |
| `scrape` | Extract elements by CSS selectors | [`/scrape`](https://developers.cloudflare.com/browser-run/quick-actions/scrape-endpoint/) |
| `discover_links` | Find all links on a page | [`/links`](https://developers.cloudflare.com/browser-run/quick-actions/links-endpoint/) |
| `screenshot` | Capture a screenshot (base64 PNG) | [`/screenshot`](https://developers.cloudflare.com/browser-run/quick-actions/screenshot-endpoint/) |

---

## RAG with Vectorize

Semantic search over your knowledge base using Cloudflare Vectorize and Workers AI embeddings.

```python
from pydantic_ai import Agent
from pydantic_ai_cloudflare import BrowserRunToolset, VectorizeToolset

# Tools for the agent: browse the web + search knowledge base
browser = BrowserRunToolset(tools=["browse"])
knowledge = VectorizeToolset(index_name="my-knowledge-base")

agent = Agent(
    "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    toolsets=[browser, knowledge],
    system_prompt=(
        "You are a research assistant. Use browse to read web pages "
        "and store_knowledge to save important findings. Use "
        "search_knowledge to find information you've previously stored."
    ),
)

result = agent.run_sync(
    "Read https://developers.cloudflare.com/workers-ai/ and store "
    "the key points. Then tell me what Workers AI can do."
)
```

The full Cloudflare-native RAG pipeline:

```
Browser Run (crawl) → Workers AI (embed) → Vectorize (store) → Workers AI (query)
```

No Pinecone. No OpenAI embeddings. No external dependencies.

---

## Conversation Persistence with D1

Conversations survive across sessions using Cloudflare D1 (serverless SQLite).

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_cloudflare import D1MessageHistory

history = D1MessageHistory(database_id="your-d1-database-id")

async def chat(session_id: str, message: str) -> str:
    agent = Agent("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast")

    # Load previous conversation
    messages = await history.get_messages(session_id)

    # Run agent with history
    result = await agent.run(message, message_history=messages)

    # Save updated conversation
    await history.save_messages(session_id, result.all_messages())
    return result.output

# First session
asyncio.run(chat("user-123", "My name is Alice and I work at Cloudflare"))
# Second session (remembers context)
asyncio.run(chat("user-123", "Where do I work?"))
# → "You work at Cloudflare"
```

---

## Observability — Zero Config

When you use `CloudflareProvider`, every LLM call routes through [AI Gateway](https://developers.cloudflare.com/ai-gateway/) automatically. You get logging, cost tracking, and analytics **without writing any extra code**.

```python
from pydantic_ai_cloudflare import GatewayObservability

obs = GatewayObservability()

# Query logs
logs = await obs.get_logs(limit=10, model="llama-3.3-70b")
for log in logs:
    print(f"{log['model']}: {log['tokens_in']}→{log['tokens_out']} tokens, cached={log['cached']}")

# Add feedback to a log entry
await obs.add_feedback(logs[0]["id"], score=95, feedback=1)

# Get analytics
analytics = await obs.get_analytics(start="2026-04-01T00:00:00Z")
```

**What you get for free:**
- Every request logged with full metadata
- Token usage and cost per request
- Cache hit/miss tracking
- Custom metadata filtering (trace IDs, session IDs, user IDs)
- Dashboard at [dash.cloudflare.com](https://dash.cloudflare.com) → AI → AI Gateway

---

## Code Mode (with Monty)

Works with [PydanticAI Code Mode](https://ai.pydantic.dev/capabilities/code-mode/). Instead of sequential tool calls, the LLM writes one Python script that calls tools in parallel. ~80% fewer LLM round-trips.

```bash
pip install 'pydantic-ai-harness[code-mode]'
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from pydantic_ai_cloudflare import BrowserRunToolset

agent = Agent(
    "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    capabilities=[CodeMode()],
    toolsets=[BrowserRunToolset()],
)

result = agent.run_sync(
    "Compare pricing on cloudflare.com/plans and aws.amazon.com/lambda/pricing"
)
# LLM writes Python that browses both pages in parallel,
# extracts pricing, and compares them — all in ONE model call.
```

---

## Explicit Configuration

### Custom provider settings

```python
from pydantic_ai_cloudflare import CloudflareProvider

provider = CloudflareProvider(
    account_id="your-account-id",
    api_key="your-api-token",
    gateway_id="production",                     # Custom AI Gateway
    gateway_metadata={"team": "data-science"},   # Custom metadata on every request
)
```

### Bypass AI Gateway

```python
provider = CloudflareProvider(
    account_id="...", api_key="...",
    gateway_id=None,  # Direct to Workers AI, no gateway
)
```

### Choose which Browser Run tools to expose

```python
from pydantic_ai_cloudflare import BrowserRunToolset

# Only browse + extract (no crawl, scrape, links, screenshot)
toolset = BrowserRunToolset(tools=["browse", "extract"])
```

### Workers AI models

Any model on [Workers AI](https://developers.cloudflare.com/workers-ai/models/) works:

```python
# Large model — best quality
"cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast"

# Reasoning model
"cloudflare:@cf/qwen/qwen3-30b-a3b"

# Vision model
"cloudflare:@cf/google/gemma-4-26b-a4b-it"

# Small + fast
"cloudflare:@cf/meta/llama-3.1-8b-instruct"
```

---

## Architecture

```
Your Python code (laptop, server, CI, notebook)
  │
  ├─ CloudflareProvider ──────────→ Workers AI ─────→ AI Gateway
  │   (LLM inference)                (free tier)       (auto logging)
  │
  ├─ BrowserRunToolset ──────────→ Browser Run
  │   (browse, extract, crawl)       (headless Chrome on edge)
  │
  ├─ CloudflareEmbeddingModel ───→ Workers AI Embeddings
  │   (text → vectors)               (bge-base/large)
  │
  ├─ VectorizeToolset ───────────→ Vectorize
  │   (search + store knowledge)     (vector database)
  │
  ├─ D1MessageHistory ──────────→ D1
  │   (conversation persistence)     (serverless SQLite)
  │
  ├─ GatewayObservability ──────→ AI Gateway API
  │   (logs, cost, feedback)          (REST API)
  │
  └─ CodeMode (Monty) ──────────→ runs in-process
      (sandboxed code execution)     (<1μs startup)
```

---

## Examples

| Example | What it shows |
|---------|-------------|
| [`simple_chat.py`](examples/simple_chat.py) | Minimal agent — 5 lines |
| [`structured_output.py`](examples/structured_output.py) | Pydantic model as LLM output |
| [`research_agent.py`](examples/research_agent.py) | Browse web + extract structured data |
| [`rag_pipeline.py`](examples/rag_pipeline.py) | Crawl → embed → store → query |
| [`persistent_chat.py`](examples/persistent_chat.py) | Multi-session conversations with D1 |
| [`code_mode_research.py`](examples/code_mode_research.py) | Parallel research with Monty |

---

## Roadmap

- [x] **v0.1.0** — CloudflareProvider, BrowserRunToolset, CloudflareEmbeddingModel, VectorizeToolset, D1MessageHistory, GatewayObservability
- [ ] **v0.2.0** — Integration tests with VCR cassettes, AI Search (AutoRAG) support
- [ ] **v0.3.0** — Upstream CloudflareProvider to `pydantic/pydantic-ai`
- [ ] **v1.0.0** — Stable API, full documentation site

## License

MIT
