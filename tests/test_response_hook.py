"""Tests for Workers AI response normalization."""

from __future__ import annotations

import json

import httpx
import pytest

from pydantic_ai_cloudflare.provider import _normalize_cf_response


def _make_response(content: str | dict, status: int = 200) -> httpx.Response:
    """Build a fake httpx response for testing."""
    body = {
        "choices": [{"message": {"content": content, "role": "assistant"}}],
        "model": "test",
    }
    resp = httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "https://test.com"),
    )
    return resp


class TestResponseNormalization:
    @pytest.mark.asyncio
    async def test_dict_content_becomes_string(self) -> None:
        """Workers AI returns content as dict -- should become JSON string."""
        resp = _make_response({"name": "Stripe", "industry": "payments"})
        await _normalize_cf_response(resp)

        data = json.loads(resp.content)
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        parsed = json.loads(content)
        assert parsed["name"] == "Stripe"

    @pytest.mark.asyncio
    async def test_markdown_fences_stripped(self) -> None:
        """Workers AI wraps JSON in ```json ... ``` -- should be stripped."""
        resp = _make_response('```json\n{"name": "test"}\n```')
        await _normalize_cf_response(resp)

        data = json.loads(resp.content)
        content = data["choices"][0]["message"]["content"]
        assert not content.startswith("```")
        assert json.loads(content) == {"name": "test"}

    @pytest.mark.asyncio
    async def test_plain_json_untouched(self) -> None:
        """Normal JSON string content should pass through unchanged."""
        resp = _make_response('{"name": "test"}')
        await _normalize_cf_response(resp)

        data = json.loads(resp.content)
        content = data["choices"][0]["message"]["content"]
        assert content == '{"name": "test"}'

    @pytest.mark.asyncio
    async def test_non_200_skipped(self) -> None:
        """Error responses should not be modified."""
        resp = _make_response("error", status=500)
        original = resp.content
        await _normalize_cf_response(resp)
        assert resp.content == original

    @pytest.mark.asyncio
    async def test_non_json_skipped(self) -> None:
        """Non-JSON responses should not be modified."""
        resp = httpx.Response(
            status_code=200,
            content=b"plain text",
            headers={"content-type": "text/plain"},
            request=httpx.Request("POST", "https://test.com"),
        )
        await _normalize_cf_response(resp)
        assert resp.content == b"plain text"
