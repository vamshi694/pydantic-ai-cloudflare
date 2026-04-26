"""Tests for GatewayObservability."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai_cloudflare.gateway import GatewayObservability


class TestGatewayCreation:
    def test_creates_with_defaults(self) -> None:
        obs = GatewayObservability(account_id="abc", api_key="tok")
        assert obs._gateway_id == "default"

    def test_custom_gateway_id(self) -> None:
        obs = GatewayObservability(gateway_id="production", account_id="abc", api_key="tok")
        assert obs._gateway_id == "production"

    def test_url_construction(self) -> None:
        obs = GatewayObservability(gateway_id="my-gw", account_id="acc", api_key="tok")
        url = obs._url("logs")
        assert "/ai-gateway/gateways/my-gw/logs" in url


class TestGetLogs:
    @pytest.mark.asyncio
    async def test_get_logs_returns_list(self) -> None:
        obs = GatewayObservability(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": [
                {
                    "id": "log-1",
                    "model": "llama-3.3-70b",
                    "provider": "workers-ai",
                    "tokens_in": 50,
                    "tokens_out": 100,
                    "success": True,
                    "cached": False,
                },
                {
                    "id": "log-2",
                    "model": "llama-3.3-70b",
                    "provider": "workers-ai",
                    "tokens_in": 30,
                    "tokens_out": 80,
                    "success": True,
                    "cached": True,
                },
            ],
        }

        with patch("pydantic_ai_cloudflare.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            logs = await obs.get_logs(limit=10)

        assert len(logs) == 2
        assert logs[0]["id"] == "log-1"
        assert logs[1]["cached"] is True


class TestAddFeedback:
    @pytest.mark.asyncio
    async def test_add_feedback(self) -> None:
        obs = GatewayObservability(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("pydantic_ai_cloudflare.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.patch = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            await obs.add_feedback("log-1", score=95, feedback=1)

        call_kwargs = mock_client.patch.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["score"] == 95
        assert body["feedback"] == 1


class TestAnalytics:
    @pytest.mark.asyncio
    async def test_get_analytics(self) -> None:
        obs = GatewayObservability(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": {"total_requests": 1000, "total_tokens": 50000},
        }

        with patch("pydantic_ai_cloudflare.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            analytics = await obs.get_analytics()

        assert analytics["total_requests"] == 1000
