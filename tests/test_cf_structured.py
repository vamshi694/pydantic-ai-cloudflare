"""Tests for cf_structured() and cf_structured_sync()."""

from __future__ import annotations

import json
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from pydantic_ai_cloudflare.structured import (
    cf_structured,
)


class SimpleOutput(BaseModel):
    name: str
    score: int


class NestedOutput(BaseModel):
    class Item(BaseModel):
        text: str
        priority: Literal["HIGH", "MEDIUM", "LOW"]

    title: str
    items: list[Item]


class TestCfStructured:
    @pytest.mark.asyncio
    async def test_simple_schema_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "choices": [{"message": {"content": '{"name": "Python", "score": 9}'}}],
        }

        with patch("pydantic_ai_cloudflare.structured.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await cf_structured("Test", SimpleOutput, account_id="abc", api_key="tok")

        assert result.name == "Python"
        assert result.score == 9

    @pytest.mark.asyncio
    async def test_dict_content_handled(self) -> None:
        """Workers AI returns content as dict -- should still parse."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "choices": [{"message": {"content": {"name": "Rust", "score": 10}}}],
        }

        with patch("pydantic_ai_cloudflare.structured.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await cf_structured("Test", SimpleOutput, account_id="abc", api_key="tok")

        assert result.name == "Rust"

    @pytest.mark.asyncio
    async def test_markdown_fenced_content(self) -> None:
        """Response wrapped in markdown fences -- should extract JSON."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "choices": [{"message": {"content": '```json\n{"name": "Go", "score": 8}\n```'}}],
        }

        with patch("pydantic_ai_cloudflare.structured.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await cf_structured("Test", SimpleOutput, account_id="abc", api_key="tok")

        assert result.name == "Go"

    @pytest.mark.asyncio
    async def test_prose_wrapped_content(self) -> None:
        """Response with prose before JSON -- should extract."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "choices": [
                {"message": {"content": 'Here is the JSON:\n{"name": "Java", "score": 7}'}}
            ],
        }

        with patch("pydantic_ai_cloudflare.structured.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await cf_structured("Test", SimpleOutput, account_id="abc", api_key="tok")

        assert result.name == "Java"

    @pytest.mark.asyncio
    async def test_retries_on_validation_error(self) -> None:
        """Should retry when first response fails validation."""
        bad_resp = MagicMock()
        bad_resp.raise_for_status = MagicMock()
        bad_resp.json.return_value = {
            "success": True,
            "choices": [{"message": {"content": '{"wrong_field": "oops"}'}}],
        }

        good_resp = MagicMock()
        good_resp.raise_for_status = MagicMock()
        good_resp.json.return_value = {
            "success": True,
            "choices": [{"message": {"content": '{"name": "Fixed", "score": 5}'}}],
        }

        with patch("pydantic_ai_cloudflare.structured.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[bad_resp, good_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await cf_structured(
                "Test", SimpleOutput, account_id="abc", api_key="tok", retries=1
            )

        assert result.name == "Fixed"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_nested_schema_with_literals(self) -> None:
        """Complex schema with Literal types works."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "title": "Test Report",
                                "items": [
                                    {"text": "Item 1", "priority": "HIGH"},
                                    {"text": "Item 2", "priority": "LOW"},
                                ],
                            }
                        )
                    }
                }
            ],
        }

        with patch("pydantic_ai_cloudflare.structured.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            result = await cf_structured("Test", NestedOutput, account_id="abc", api_key="tok")

        assert result.title == "Test Report"
        assert len(result.items) == 2
        assert result.items[0].priority == "HIGH"

    @pytest.mark.asyncio
    async def test_sends_schema_in_system_prompt(self) -> None:
        """Verify the schema is injected into the system message."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "choices": [{"message": {"content": '{"name": "test", "score": 1}'}}],
        }

        with patch("pydantic_ai_cloudflare.structured.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            await cf_structured("Test", SimpleOutput, account_id="abc", api_key="tok")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        system_msg = body["messages"][0]["content"]
        assert "JSON Schema" in system_msg
        assert "name" in system_msg
        assert "score" in system_msg
        assert body["response_format"]["type"] == "json_object"
