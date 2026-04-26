"""Cloudflare Workers AI provider for PydanticAI.

Routes through AI Gateway by default for zero-config observability
(logging, cost tracking, analytics).

Usage::

    from pydantic_ai import Agent

    # Zero-config: env vars + auto AI Gateway
    agent = Agent("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast")

    # Explicit configuration
    from pydantic_ai_cloudflare import CloudflareProvider
    provider = CloudflareProvider(
        account_id="abc123",
        api_key="my-token",
        gateway_id="production",
    )
    agent = Agent("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast", provider=provider)

    result = agent.run_sync("What is Cloudflare?")
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models import ModelProfile
from pydantic_ai.providers import Provider

from ._auth import resolve_account_id, resolve_api_token
from .profiles import cloudflare_model_profile

if TYPE_CHECKING:
    pass


class CloudflareProvider(Provider[AsyncOpenAI]):
    """PydanticAI provider for Cloudflare Workers AI.

    Workers AI exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint,
    so this provider reuses ``AsyncOpenAI`` as its SDK client.

    By default, all requests route through **AI Gateway** (``gateway_id="default"``),
    which gives you logging, analytics, and cost tracking with zero extra code.
    The ``default`` gateway auto-creates on first authenticated request.

    Args:
        account_id: Cloudflare account ID. Falls back to ``CLOUDFLARE_ACCOUNT_ID``
            or ``CF_ACCOUNT_ID`` env var.
        api_key: Cloudflare API token. Falls back to ``CLOUDFLARE_API_TOKEN``,
            ``CF_API_TOKEN``, or ``CF_AI_API_TOKEN`` env var.
        gateway_id: AI Gateway ID. ``"default"`` auto-creates a gateway.
            Set to ``None`` to bypass the gateway entirely.
        gateway_metadata: Custom metadata (max 5 key-value pairs) attached to
            every request via AI Gateway. Useful for trace IDs, session IDs,
            user IDs. Values must be str, int, float, or bool.
        http_client: Optional pre-configured httpx.AsyncClient.
    """

    _account_id: str
    _api_key: str
    _gateway_id: str | None
    _gateway_metadata: dict[str, str | int | float | bool] | None

    def __init__(
        self,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        gateway_id: str | None = "default",
        gateway_metadata: dict[str, str | int | float | bool] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._gateway_id = gateway_id
        self._gateway_metadata = gateway_metadata

        # Build extra headers for AI Gateway
        extra_headers: dict[str, str] = {}
        if self._gateway_id is not None:
            extra_headers["cf-aig-authorization"] = f"Bearer {self._api_key}"
            if self._gateway_metadata:
                extra_headers["cf-aig-metadata"] = json.dumps(self._gateway_metadata)

        if http_client is not None:
            openai_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                http_client=http_client,
                default_headers=extra_headers or None,
            )
            self._client = openai_client
        else:
            own_http = httpx.AsyncClient(timeout=60.0)
            openai_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                http_client=own_http,
                default_headers=extra_headers or None,
            )
            self._client = openai_client
            self._own_http_client = own_http

    @property
    def name(self) -> str:
        """Provider name for PydanticAI's model registry."""
        return "cloudflare"

    @property
    def base_url(self) -> str:
        """Workers AI endpoint, optionally routed through AI Gateway."""
        if self._gateway_id is not None:
            return (
                f"https://gateway.ai.cloudflare.com/v1/"
                f"{self._account_id}/{self._gateway_id}/workers-ai/v1"
            )
        return f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/v1"

    @property
    def client(self) -> AsyncOpenAI:
        """The OpenAI-compatible SDK client."""
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """Return capabilities profile for the given Workers AI model."""
        return cloudflare_model_profile(model_name)
