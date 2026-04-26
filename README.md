# pydantic-ai-cloudflare

PydanticAI integration for Cloudflare's AI stack.

Build AI agents with **type-safe structured output**, **web browsing**, and **zero-config observability** -- entirely on Cloudflare's free tier.

```bash
pip install pydantic-ai-cloudflare
```

## Quick Start

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class Answer(BaseModel):
    response: str
    confidence: float

agent = Agent(
    "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    output_type=Answer,
)

result = agent.run_sync("What is Cloudflare?")
print(result.output.response)    # typed string, not raw text
print(result.output.confidence)  # validated float
```

Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` environment variables. That's it.

## What's Included

### Workers AI Model Provider

Use any [Workers AI model](https://developers.cloudflare.com/workers-ai/models/) as a PydanticAI model. Tool calling, streaming, structured output -- all via the OpenAI-compatible API.

```python
from pydantic_ai import Agent

# Llama 3.3 70B -- free tier
agent = Agent("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast")

# Qwen 3 30B -- with reasoning/thinking
agent = Agent("cloudflare:@cf/qwen/qwen3-30b-a3b")

# Any Workers AI model
agent = Agent("cloudflare:@cf/google/gemma-4-26b-a4b-it")
```

**AI Gateway is on by default.** Every LLM call is automatically logged with cost tracking and analytics at [dash.cloudflare.com](https://dash.cloudflare.com). No extra code.

### Browser Run Toolset

Give your agents the ability to browse the web via [Cloudflare Browser Run](https://developers.cloudflare.com/browser-run/). Renders JavaScript-heavy pages on Cloudflare's edge -- no local browser needed.

```python
from pydantic_ai import Agent
from pydantic_ai_cloudflare import BrowserRunToolset

agent = Agent(
    "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    toolsets=[BrowserRunToolset()],
)
result = agent.run_sync("What's on the Cloudflare pricing page?")
```

Tools available to the agent:

| Tool | What it does |
|------|-------------|
| `browse` | Fetch any page as clean markdown |
| `extract` | AI-powered structured data extraction from a page |
| `crawl` | Crawl an entire site (async, configurable depth/limit) |
| `scrape` | Extract specific elements with CSS selectors |
| `discover_links` | Find all links on a page |
| `screenshot` | Capture a screenshot (base64 PNG) |

### Structured Output

The reason Python AI engineers use Pydantic. Your LLM output is a **validated Python object**, not a string.

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
# result.output is a validated PricingAnalysis instance
# result.output.plans[0].name → "Free"
# result.output.plans[0].price → "$0"
# result.output.has_free_tier → True
```

### Code Mode (with Monty)

Works with [PydanticAI Code Mode](https://ai.pydantic.dev/capabilities/code-mode/) out of the box. The LLM writes Python code that calls your tools in parallel -- Monty executes it safely in a sandbox.

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
# The LLM writes ONE Python script that browses both pages in parallel
# using asyncio.gather, extracts pricing, and compares them.
# ~80% fewer LLM round-trips than sequential tool calling.
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CLOUDFLARE_ACCOUNT_ID` | Yes | Your Cloudflare account ID |
| `CLOUDFLARE_API_TOKEN` | Yes | API token ([create one](https://dash.cloudflare.com/profile/api-tokens)) |

Also supports `CF_ACCOUNT_ID`, `CF_API_TOKEN`, `CF_AI_API_TOKEN` as fallbacks.

The API token needs these permissions:
- **Workers AI** -- Read (for model inference)
- **Browser Rendering** -- Edit (for Browser Run tools)

### Explicit Configuration

```python
from pydantic_ai_cloudflare import CloudflareProvider

provider = CloudflareProvider(
    account_id="your-account-id",
    api_key="your-api-token",
    gateway_id="production",        # AI Gateway ID (default: "default")
    gateway_metadata={               # Custom metadata for observability
        "session_id": "sess-123",
        "user_id": "u-456",
    },
)
```

### Disable AI Gateway

```python
provider = CloudflareProvider(
    account_id="...",
    api_key="...",
    gateway_id=None,  # Direct Workers AI, no gateway
)
```

## Architecture

```
Your Python code (runs anywhere)
  │
  ├─ CloudflareProvider ────→ Workers AI (inference)
  │   via AI Gateway           └→ logging, cost tracking, analytics (free)
  │
  ├─ BrowserRunToolset ────→ Browser Run (web interaction)
  │                            └→ /markdown, /json, /crawl, /scrape, /links
  │
  └─ CodeMode (Monty) ────→ runs locally in-process
                              └→ sandboxed Python execution (<1μs startup)
```

Cloudflare handles the heavy lifting (inference, web rendering). Monty handles the lightweight part (executing LLM-generated code). Your code runs anywhere -- laptop, server, CI, notebook.

## Examples

See the [`examples/`](examples/) directory:

- [`simple_chat.py`](examples/simple_chat.py) -- Minimal agent
- [`structured_output.py`](examples/structured_output.py) -- Pydantic validated output
- [`research_agent.py`](examples/research_agent.py) -- Web research with Browser Run
- [`code_mode_research.py`](examples/code_mode_research.py) -- Parallel research with Code Mode

## Roadmap

| Version | What's coming |
|---------|--------------|
| **v0.1.0** | CloudflareProvider + BrowserRunToolset (current) |
| **v0.2.0** | VectorizeToolset + CloudflareEmbedder (RAG) |
| **v0.3.0** | D1MessageHistory (conversation persistence) |
| **v0.4.0** | AI Gateway observability APIs (logs, cost, feedback) |

## License

MIT
