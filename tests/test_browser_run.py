"""Tests for BrowserRunToolset."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai_cloudflare._errors import CloudflareAPIError, CloudflareConfigError
from pydantic_ai_cloudflare._http import check_api_response
from pydantic_ai_cloudflare.browser_run import BrowserRunToolset


class TestToolsetCreation:
    """Verify toolset initialization and configuration."""

    def test_creates_with_credentials(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")
        assert toolset._account_id == "abc"

    def test_missing_account_id_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(CloudflareConfigError, match="account ID"):
                BrowserRunToolset(account_id=None, api_key="tok")

    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(CloudflareConfigError, match="API token"):
                BrowserRunToolset(account_id="abc", api_key=None)

    def test_default_tools_all_enabled(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")
        assert len(toolset._enabled_tools) == 6

    def test_custom_tool_subset(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok", tools=["browse", "extract"])
        assert toolset._enabled_tools == frozenset({"browse", "extract"})

    def test_toolset_id(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")
        assert toolset.id == "browser_run"

    def test_url_construction(self) -> None:
        toolset = BrowserRunToolset(account_id="my-acct", api_key="tok")
        assert toolset._url("markdown") == (
            "https://api.cloudflare.com/client/v4/accounts/my-acct/browser-rendering/markdown"
        )


class TestErrorHandling:
    """Verify API error envelope detection."""

    def test_success_false_raises(self) -> None:
        with pytest.raises(CloudflareAPIError, match="Cloudflare API error"):
            check_api_response({"success": False, "errors": [{"message": "bad request"}]})

    def test_success_true_passes(self) -> None:
        check_api_response({"success": True, "result": "ok"})

    def test_non_dict_passes(self) -> None:
        check_api_response("plain string")
        check_api_response(["a", "list"])


class TestBrowseTool:
    """Verify the browse (markdown) tool."""

    @pytest.mark.asyncio
    async def test_browse_returns_markdown(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": "# Example\n\nHello world",
        }

        with patch("pydantic_ai_cloudflare.browser_run.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await toolset._browse("https://example.com")

        assert result == "# Example\n\nHello world"

    @pytest.mark.asyncio
    async def test_browse_api_error_raises(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": False,
            "errors": [{"message": "invalid URL"}],
        }

        with patch("pydantic_ai_cloudflare.browser_run.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with pytest.raises(CloudflareAPIError):
                await toolset._browse("https://bad-url")


class TestExtractTool:
    """Verify the extract (json) tool."""

    @pytest.mark.asyncio
    async def test_extract_returns_json(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": {"company": "Cloudflare", "industry": "Technology"},
        }

        with patch("pydantic_ai_cloudflare.browser_run.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await toolset._extract("https://example.com", "Extract company info")

        parsed = json.loads(result)
        assert parsed["company"] == "Cloudflare"


class TestScrapeTool:
    """Verify the scrape tool."""

    @pytest.mark.asyncio
    async def test_scrape_returns_elements(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": [
                {
                    "selector": "h1",
                    "results": [{"text": "Welcome"}],
                },
                {
                    "selector": ".price",
                    "results": [{"text": "$20/mo"}, {"text": "$200/mo"}],
                },
            ],
        }

        with patch("pydantic_ai_cloudflare.browser_run.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await toolset._scrape("https://example.com", ["h1", ".price"])

        assert "Welcome" in result
        assert "$20/mo" in result


class TestScreenshotTool:
    """Verify the screenshot tool's binary error detection."""

    @pytest.mark.asyncio
    async def test_screenshot_json_error_raises(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "success": False,
            "errors": [{"message": "failed"}],
        }

        with patch("pydantic_ai_cloudflare.browser_run.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with pytest.raises(CloudflareAPIError):
                await toolset._screenshot("https://example.com")

    @pytest.mark.asyncio
    async def test_screenshot_binary_success(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.content = b"\x89PNG\r\n\x1a\nfake"

        with patch("pydantic_ai_cloudflare.browser_run.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await toolset._screenshot("https://example.com")

        assert isinstance(result, str)
        assert len(result) > 0


class TestDiscoverLinksTool:
    """Verify the discover_links tool."""

    @pytest.mark.asyncio
    async def test_returns_links(self) -> None:
        toolset = BrowserRunToolset(account_id="abc", api_key="tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": [
                "https://example.com/about",
                "https://example.com/pricing",
            ],
        }

        with patch("pydantic_ai_cloudflare.browser_run.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await toolset._discover_links("https://example.com")

        assert "https://example.com/about" in result
        assert "https://example.com/pricing" in result
