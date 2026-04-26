"""Cloudflare Workers AI provider for PydanticAI.

Routes through AI Gateway by default for zero-config observability.
"""

from __future__ import annotations

import json

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models import ModelProfile
from pydantic_ai.providers import Provider

from ._auth import resolve_account_id, resolve_api_token
from .profiles import cloudflare_model_profile


class CloudflareProvider(Provider[AsyncOpenAI]):
    """Workers AI via OpenAI-compatible API, with AI Gateway auto-routing.

    By default requests go through AI Gateway (gateway_id="default") which
    gives you logging, analytics, and cost tracking for free.
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

        # AI Gateway headers -- cf-aig-authorization is separate from the
        # provider auth, and metadata gets attached to every log entry.
        extra_headers: dict[str, str] = {}
        if self._gateway_id is not None:
            extra_headers["cf-aig-authorization"] = f"Bearer {self._api_key}"
            if self._gateway_metadata:
                extra_headers["cf-aig-metadata"] = json.dumps(self._gateway_metadata)

        if http_client is not None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                http_client=http_client,
                default_headers=extra_headers or None,
            )
        else:
            own_http = httpx.AsyncClient(timeout=60.0)
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                http_client=own_http,
                default_headers=extra_headers or None,
            )
            self._own_http_client = own_http

    @property
    def name(self) -> str:
        return "cloudflare"

    @property
    def base_url(self) -> str:
        if self._gateway_id is not None:
            return (
                f"https://gateway.ai.cloudflare.com/v1/"
                f"{self._account_id}/{self._gateway_id}/workers-ai/v1"
            )
        return f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/v1"

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return cloudflare_model_profile(model_name)
