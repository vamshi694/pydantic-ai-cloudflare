"""PydanticAI integration for Cloudflare's AI stack."""

from __future__ import annotations

from .agent import cloudflare_agent
from .browser_run import BrowserRunToolset
from .d1 import D1MessageHistory
from .embeddings import CloudflareEmbeddingModel
from .gateway import GatewayObservability
from .models import list_models, recommend_model
from .provider import CloudflareProvider, cloudflare_model
from .structured import extract_json_from_text, schema_stats, simplify_schema
from .vectorize import VectorizeToolset

__all__ = [
    # Core
    "CloudflareProvider",
    "cloudflare_model",
    "cloudflare_agent",
    # Model discovery
    "list_models",
    "recommend_model",
    # Tools
    "BrowserRunToolset",
    "VectorizeToolset",
    # Embeddings
    "CloudflareEmbeddingModel",
    # Persistence
    "D1MessageHistory",
    # Observability
    "GatewayObservability",
    # Schema utilities
    "simplify_schema",
    "schema_stats",
    "extract_json_from_text",
]

__version__ = "0.1.0"
