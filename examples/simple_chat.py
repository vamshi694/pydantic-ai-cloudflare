"""Simplest possible agent using Cloudflare Workers AI.

Set environment variables:
    CLOUDFLARE_ACCOUNT_ID=your-account-id
    CLOUDFLARE_API_TOKEN=your-api-token

Run:
    uv run python examples/simple_chat.py
"""

from pydantic_ai import Agent

from pydantic_ai_cloudflare import cloudflare_model

agent = Agent(
    cloudflare_model(),
    system_prompt="You are a helpful assistant. Be concise.",
)

result = agent.run_sync("What is Cloudflare Workers AI in one sentence?")
print(result.output)
