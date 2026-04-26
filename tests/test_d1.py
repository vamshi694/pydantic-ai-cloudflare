"""Tests for D1MessageHistory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai_cloudflare.d1 import D1MessageHistory


class TestD1Creation:
    def test_creates_with_config(self) -> None:
        h = D1MessageHistory(database_id="db-123", account_id="abc", api_key="tok")
        assert h._database_id == "db-123"
        assert h._table_name == "pydantic_ai_messages"

    def test_custom_table_name(self) -> None:
        h = D1MessageHistory(
            database_id="db", account_id="abc", api_key="tok", table_name="my_msgs"
        )
        assert h._table_name == "my_msgs"

    def test_url_construction(self) -> None:
        h = D1MessageHistory(database_id="db-456", account_id="acc", api_key="tok")
        url = h._url("query")
        assert "/d1/database/db-456/query" in url


class TestGetMessages:
    @pytest.mark.asyncio
    async def test_get_messages_empty(self) -> None:
        h = D1MessageHistory(database_id="db", account_id="abc", api_key="tok")

        create_resp = MagicMock()
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {"success": True, "result": [{}]}

        query_resp = MagicMock()
        query_resp.raise_for_status = MagicMock()
        query_resp.json.return_value = {
            "success": True,
            "result": [{"results": []}],
        }

        with patch("pydantic_ai_cloudflare.d1.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[create_resp, create_resp, query_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            messages = await h.get_messages("sess-1")

        assert messages == []

    @pytest.mark.asyncio
    async def test_get_messages_with_data(self) -> None:
        h = D1MessageHistory(
            database_id="db", account_id="abc", api_key="tok", auto_create_table=False
        )
        h._table_created = True

        query_resp = MagicMock()
        query_resp.raise_for_status = MagicMock()
        query_resp.json.return_value = {
            "success": True,
            "result": [
                {
                    "results": [
                        {"message_data": '{"role": "user", "content": "hello"}'},
                        {"message_data": '{"role": "assistant", "content": "hi"}'},
                    ]
                }
            ],
        }

        with patch("pydantic_ai_cloudflare.d1.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=query_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            messages = await h.get_messages("sess-1")

        assert len(messages) == 2


class TestSaveMessages:
    @pytest.mark.asyncio
    async def test_save_messages(self) -> None:
        h = D1MessageHistory(
            database_id="db", account_id="abc", api_key="tok", auto_create_table=False
        )
        h._table_created = True

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"success": True, "result": [{}]}

        with patch("pydantic_ai_cloudflare.d1.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            await h.save_messages(
                "sess-1",
                [{"role": "user", "content": "hello"}],
            )

        # DELETE + 1 INSERT = 2 calls
        assert mock_client.post.call_count == 2


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions(self) -> None:
        h = D1MessageHistory(
            database_id="db", account_id="abc", api_key="tok", auto_create_table=False
        )
        h._table_created = True

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "success": True,
            "result": [
                {
                    "results": [
                        {"session_id": "s1", "message_count": 5, "last_updated": "2026-04-26"},
                        {"session_id": "s2", "message_count": 3, "last_updated": "2026-04-25"},
                    ]
                }
            ],
        }

        with patch("pydantic_ai_cloudflare.d1.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            sessions = await h.list_sessions()

        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "s1"
