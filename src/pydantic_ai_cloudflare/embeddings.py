"""Workers AI embedding model for PydanticAI's Embedder system."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from pydantic_ai.embeddings.base import EmbeddingModel, EmbedInputType
from pydantic_ai.embeddings.result import EmbeddingResult
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_ai.usage import RequestUsage

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response

# Default embedding model on Workers AI
DEFAULT_EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5"
DEFAULT_DIMENSIONS = 768


class CloudflareEmbeddingModel(EmbeddingModel):
    """Workers AI embedding model for PydanticAI.

    Calls the Workers AI ``/v1/embeddings`` endpoint (OpenAI-compatible)
    to generate text embeddings.

    Args:
        model_name: Workers AI embedding model ID.
            Default: ``@cf/baai/bge-base-en-v1.5`` (768 dimensions).
        account_id: Cloudflare account ID. Falls back to env vars.
        api_key: Cloudflare API token. Falls back to env vars.
        gateway_id: AI Gateway ID for routing. ``"default"`` auto-creates.
            Set to ``None`` to bypass gateway.
        dimensions: Embedding dimensions. Default depends on model.
        request_timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        gateway_id: str | None = "default",
        dimensions: int = DEFAULT_DIMENSIONS,
        request_timeout: float = DEFAULT_TIMEOUT,
        settings: EmbeddingSettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._model_name = model_name
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._gateway_id = gateway_id
        self._dimensions = dimensions
        self._timeout = request_timeout
        self._headers = build_headers(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "cloudflare"

    @property
    def base_url(self) -> str | None:
        if self._gateway_id is not None:
            return (
                f"https://gateway.ai.cloudflare.com/v1/"
                f"{self._account_id}/{self._gateway_id}/workers-ai/v1"
            )
        return f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/v1"

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        """Generate embeddings for the given inputs.

        Args:
            inputs: Text string or list of strings to embed.
            input_type: Whether these are queries or documents.
            settings: Optional embedding settings override.

        Returns:
            EmbeddingResult with embedding vectors.
        """
        input_list, settings = self.prepare_embed(inputs, settings)

        url = f"{self.base_url}/embeddings"
        body: dict[str, Any] = {
            "input": input_list,
            "model": self._model_name,
        }

        dims = (settings or {}).get("dimensions") or self._dimensions
        if dims:
            body["dimensions"] = dims

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=self._headers, json=body)
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)

        embeddings_data = data.get("data", [])
        embeddings = [item["embedding"] for item in embeddings_data]

        usage_data = data.get("usage", {})
        usage = RequestUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=0,
        )

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=input_list,
            input_type=input_type,
            usage=usage,
            model_name=data.get("model", self._model_name),
            provider_name="cloudflare",
        )
