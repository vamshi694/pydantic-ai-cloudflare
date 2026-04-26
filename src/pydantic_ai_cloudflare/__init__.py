"""PydanticAI integration for Cloudflare's AI stack.

Provides Workers AI model inference, Browser Run web tools,
and zero-config observability via AI Gateway.

Quick start::

    from pydantic_ai import Agent

    agent = Agent("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    result = agent.run_sync("Hello!")
"""

from __future__ import annotations

from .browser_run import BrowserRunToolset
from .provider import CloudflareProvider

__all__ = [
    "CloudflareProvider",
    "BrowserRunToolset",
]

__version__ = "0.1.0"
