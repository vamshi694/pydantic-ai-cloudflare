"""PydanticAI integration for Cloudflare's AI stack.

Build AI agents with type-safe structured output, web browsing, RAG,
conversation persistence, and zero-config observability — entirely
on Cloudflare's infrastructure.

Quick start::

    from pydantic_ai import Agent

    agent = Agent("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    result = agent.run_sync("Hello!")
"""

from __future__ import annotations

from .browser_run import BrowserRunToolset
from .d1 import D1MessageHistory
from .embeddings import CloudflareEmbeddingModel
from .gateway import GatewayObservability
from .provider import CloudflareProvider
from .vectorize import VectorizeToolset

__all__ = [
    # Core
    "CloudflareProvider",
    "CloudflareEmbeddingModel",
    # Tools
    "BrowserRunToolset",
    "VectorizeToolset",
    # Persistence
    "D1MessageHistory",
    # Observability
    "GatewayObservability",
]

__version__ = "0.1.0"
