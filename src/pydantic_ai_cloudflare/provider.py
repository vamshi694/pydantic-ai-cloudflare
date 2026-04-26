"""Cloudflare Workers AI provider for PydanticAI.

Routes through AI Gateway by default for zero-config observability.
"""

from __future__ import annotations

import json as json_mod
import re
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models import ModelProfile
from pydantic_ai.providers import Provider

from ._auth import resolve_account_id, resolve_api_token
from .profiles import cloudflare_model_profile


async def _normalize_cf_response(response: httpx.Response) -> None:
    """Fix Workers AI response quirks before the OpenAI SDK parses them.

    Workers AI has two incompatibilities with the OpenAI spec:
    1. Returns message.content as a dict (parsed JSON) instead of a string
       when using response_format: json_object
    2. Sometimes wraps JSON in markdown code fences (```json ... ```)

    This event hook rewrites the raw HTTP response body to fix both issues.
    """
    if response.status_code != 200:
        return

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return

    # Need to read the body before we can inspect/modify it
    await response.aread()

    try:
        data = response.json()
    except Exception:
        return

    modified = False
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content")

        # Fix 1: dict content → JSON string
        if isinstance(content, dict):
            msg["content"] = json_mod.dumps(content)
            modified = True
        elif isinstance(content, str):
            stripped = content.strip()
            needs_fix = False

            # Fix 2: strip markdown code fences
            if "```" in stripped:
                fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
                if fenced:
                    stripped = fenced.group(1).strip()
                    needs_fix = True

            # Fix 3: extract JSON from prose ("Here is the JSON: {...}")
            if not stripped.startswith("{") and not stripped.startswith("["):
                first_brace = stripped.find("{")
                last_brace = stripped.rfind("}")
                if first_brace != -1 and last_brace > first_brace:
                    candidate = stripped[first_brace : last_brace + 1]
                    try:
                        json_mod.loads(candidate)
                        stripped = candidate
                        needs_fix = True
                    except json_mod.JSONDecodeError:
                        pass

            if needs_fix:
                msg["content"] = stripped
                modified = True

    if modified:
        # Rewrite the response body so the OpenAI SDK gets clean data
        new_body = json_mod.dumps(data).encode("utf-8")
        response._content = new_body
        response.headers["content-length"] = str(len(new_body))


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
                extra_headers["cf-aig-metadata"] = json_mod.dumps(self._gateway_metadata)

        if http_client is not None:
            # User-provided client -- add our response hook
            http_client.event_hooks.setdefault("response", []).append(_normalize_cf_response)
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                http_client=http_client,
                default_headers=extra_headers or None,
            )
        else:
            own_http = httpx.AsyncClient(
                timeout=60.0,
                event_hooks={"response": [_normalize_cf_response]},
            )
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

    def chat_model(self, model_name: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast") -> Any:
        """Create an OpenAIChatModel wired to this provider.

        Convenience method so you don't have to import OpenAIChatModel yourself.

        Usage::

            provider = CloudflareProvider()
            agent = Agent(provider.chat_model("@cf/meta/llama-3.3-70b-instruct-fp8-fast"))
        """
        from pydantic_ai.models.openai import OpenAIChatModel

        return OpenAIChatModel(model_name, provider=self)


def cloudflare_model(
    model_name: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    *,
    account_id: str | None = None,
    api_key: str | None = None,
    gateway_id: str | None = "default",
    gateway_metadata: dict[str, str | int | float | bool] | None = None,
) -> Any:
    """Create a PydanticAI model backed by Cloudflare Workers AI.

    This is the simplest way to use Workers AI with PydanticAI::

        from pydantic_ai import Agent
        from pydantic_ai_cloudflare import cloudflare_model

        agent = Agent(cloudflare_model())

    Or with a specific model::

        agent = Agent(cloudflare_model("@cf/qwen/qwen3-30b-a3b"))
    """
    from pydantic_ai.models.openai import OpenAIChatModel

    provider = CloudflareProvider(
        account_id=account_id,
        api_key=api_key,
        gateway_id=gateway_id,
        gateway_metadata=gateway_metadata,
    )
    return OpenAIChatModel(model_name, provider=provider)
