"""Tests for VectorizeToolset."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai_cloudflare.vectorize import VectorizeToolset


class TestVectorizeCreation:
    def test_creates_with_config(self) -> None:
        ts = VectorizeToolset(index_name="my-idx", account_id="abc", api_key="tok")
        assert ts._index_name == "my-idx"
        assert ts.id == "vectorize"

    def test_default_embedding_model(self) -> None:
        ts = VectorizeToolset(index_name="idx", account_id="abc", api_key="tok")
        assert "bge-base" in ts._embedding_model

    def test_url_construction(self) -> None:
        ts = VectorizeToolset(index_name="docs", account_id="acc", api_key="tok")
        url = ts._vectorize_url("query")
        assert "/vectorize/v2/indexes/docs/query" in url


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        ts = VectorizeToolset(index_name="idx", account_id="abc", api_key="tok")

        embed_resp = MagicMock()
        embed_resp.raise_for_status = MagicMock()
        embed_resp.json.return_value = {
            "success": True,
            "result": {"data": [[0.1, 0.2, 0.3]]},
        }

        search_resp = MagicMock()
        search_resp.raise_for_status = MagicMock()
        search_resp.json.return_value = {
            "success": True,
            "result": {
                "matches": [
                    {
                        "score": 0.95,
                        "metadata": {"text": "Cloudflare is great", "source": "docs"},
                    },
                    {
                        "score": 0.82,
                        "metadata": {"text": "Workers AI rocks", "source": "blog"},
                    },
                ]
            },
        }

        with patch("pydantic_ai_cloudflare.vectorize.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embed_resp, search_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await ts._search("What is Cloudflare?")

        assert "Cloudflare is great" in result
        assert "0.950" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self) -> None:
        ts = VectorizeToolset(index_name="idx", account_id="abc", api_key="tok")

        embed_resp = MagicMock()
        embed_resp.raise_for_status = MagicMock()
        embed_resp.json.return_value = {
            "success": True,
            "result": {"data": [[0.1, 0.2]]},
        }

        search_resp = MagicMock()
        search_resp.raise_for_status = MagicMock()
        search_resp.json.return_value = {
            "success": True,
            "result": {"matches": []},
        }

        with patch("pydantic_ai_cloudflare.vectorize.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embed_resp, search_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await ts._search("nonexistent")

        assert "No relevant documents" in result


class TestStore:
    @pytest.mark.asyncio
    async def test_store_success(self) -> None:
        ts = VectorizeToolset(index_name="idx", account_id="abc", api_key="tok")

        embed_resp = MagicMock()
        embed_resp.raise_for_status = MagicMock()
        embed_resp.json.return_value = {
            "success": True,
            "result": {"data": [[0.1, 0.2, 0.3]]},
        }

        upsert_resp = MagicMock()
        upsert_resp.raise_for_status = MagicMock()
        upsert_resp.json.return_value = {"success": True, "result": {}}

        with patch("pydantic_ai_cloudflare.vectorize.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[embed_resp, upsert_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await ts._store("Hello world", source="test")

        assert "Stored document" in result
