"""Tests for CloudflareEmbeddingModel."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai_cloudflare.embeddings import CloudflareEmbeddingModel


class TestEmbeddingCreation:
    def test_creates_with_defaults(self) -> None:
        model = CloudflareEmbeddingModel(account_id="abc", api_key="tok")
        assert model.model_name == "@cf/baai/bge-base-en-v1.5"
        assert model.system == "cloudflare"

    def test_custom_model(self) -> None:
        model = CloudflareEmbeddingModel(
            "@cf/baai/bge-large-en-v1.5", account_id="abc", api_key="tok"
        )
        assert model.model_name == "@cf/baai/bge-large-en-v1.5"

    def test_base_url_with_gateway(self) -> None:
        model = CloudflareEmbeddingModel(account_id="abc", api_key="tok", gateway_id="default")
        assert "gateway.ai.cloudflare.com" in (model.base_url or "")

    def test_base_url_without_gateway(self) -> None:
        model = CloudflareEmbeddingModel(account_id="abc", api_key="tok", gateway_id=None)
        assert "api.cloudflare.com" in (model.base_url or "")


class TestEmbed:
    @pytest.mark.asyncio
    async def test_embed_single_text(self) -> None:
        model = CloudflareEmbeddingModel(account_id="abc", api_key="tok", gateway_id=None)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "data": [{"embedding": [0.1, 0.2, 0.3] * 256}],
            "usage": {"prompt_tokens": 5},
            "model": "@cf/baai/bge-base-en-v1.5",
        }

        with patch("pydantic_ai_cloudflare.embeddings.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await model.embed("Hello world", input_type="query")

        assert len(result.embeddings) == 1
        assert result.provider_name == "cloudflare"

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self) -> None:
        model = CloudflareEmbeddingModel(account_id="abc", api_key="tok", gateway_id=None)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "data": [
                {"embedding": [0.1] * 768},
                {"embedding": [0.2] * 768},
            ],
            "usage": {"prompt_tokens": 10},
            "model": "@cf/baai/bge-base-en-v1.5",
        }

        with patch("pydantic_ai_cloudflare.embeddings.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await model.embed(["Hello", "World"], input_type="document")

        assert len(result.embeddings) == 2
        assert result.input_type == "document"
