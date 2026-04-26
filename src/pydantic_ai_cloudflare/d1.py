"""D1-backed message history for multi-session agents.

Stores PydanticAI ModelMessage objects in Cloudflare D1 (serverless SQLite).
Table is auto-created on first use.
    await history.save_messages("session-123", result.all_messages())
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic_ai.messages import ModelMessage

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response

logger = logging.getLogger(__name__)

D1_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    message_data TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, message_index)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_{table_name}_session
ON {table_name}(session_id, message_index);
"""


class D1MessageHistory:
    """Conversation persistence using Cloudflare D1.

    Stores PydanticAI ``ModelMessage`` objects as JSON in a D1 SQLite
    database, keyed by session ID.

    Args:
        database_id: The D1 database UUID.
        account_id: Cloudflare account ID. Falls back to env vars.
        api_key: Cloudflare API token. Falls back to env vars.
        table_name: SQL table name. Default: ``pydantic_ai_messages``.
        request_timeout: HTTP request timeout in seconds.
        auto_create_table: Create the table on first use if it doesn't exist.
    """

    def __init__(
        self,
        *,
        database_id: str,
        account_id: str | None = None,
        api_key: str | None = None,
        table_name: str = "pydantic_ai_messages",
        request_timeout: float = DEFAULT_TIMEOUT,
        auto_create_table: bool = True,
    ) -> None:
        self._database_id = database_id
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._table_name = table_name
        self._timeout = request_timeout
        self._auto_create = auto_create_table
        self._table_created = False
        self._headers = build_headers(self._api_key)

    def _url(self, path: str = "query") -> str:
        """Build the D1 API URL."""
        return f"{D1_BASE_URL}/{self._account_id}/d1/database/{self._database_id}/{path}"

    async def _execute(self, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
        """Execute a SQL statement against D1."""
        body: dict[str, Any] = {"sql": sql}
        if params:
            body["params"] = params

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("query"),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        results = data.get("result", [])
        return results[0] if results else {}

    async def _ensure_table(self) -> None:
        """Create the messages table if it doesn't exist."""
        if self._table_created or not self._auto_create:
            return

        await self._execute(CREATE_TABLE_SQL.format(table_name=self._table_name))
        await self._execute(CREATE_INDEX_SQL.format(table_name=self._table_name))
        self._table_created = True

    async def get_messages(self, session_id: str) -> list[ModelMessage]:
        """Load conversation messages for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of ModelMessage objects in order.
        """
        await self._ensure_table()

        result = await self._execute(
            f"SELECT message_data FROM {self._table_name} "
            f"WHERE session_id = ? ORDER BY message_index ASC",
            [session_id],
        )

        rows = result.get("results", [])
        messages: list[ModelMessage] = []
        for row in rows:
            data = json.loads(row["message_data"])
            # Reconstruct ModelMessage from serialized form
            messages.append(data)

        return messages

    async def save_messages(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Save conversation messages for a session.

        Replaces all existing messages for the session.

        Args:
            session_id: Unique session identifier.
            messages: List of ModelMessage objects to store.
        """
        await self._ensure_table()

        # Delete existing messages for this session
        await self._execute(
            f"DELETE FROM {self._table_name} WHERE session_id = ?",
            [session_id],
        )

        # Insert new messages
        for i, msg in enumerate(messages):
            msg_json = json.dumps(msg, default=str)
            await self._execute(
                f"INSERT INTO {self._table_name} "
                f"(session_id, message_index, message_data, created_at) "
                f"VALUES (?, ?, ?, ?)",
                [
                    session_id,
                    i,
                    msg_json,
                    datetime.now(timezone.utc).isoformat(),
                ],
            )

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all conversation sessions.

        Returns:
            List of dicts with session_id, message_count, and last_updated.
        """
        await self._ensure_table()

        result = await self._execute(
            f"SELECT session_id, COUNT(*) as message_count, "
            f"MAX(created_at) as last_updated "
            f"FROM {self._table_name} "
            f"GROUP BY session_id ORDER BY last_updated DESC",
        )

        return result.get("results", [])

    async def delete_session(self, session_id: str) -> None:
        """Delete all messages for a session.

        Args:
            session_id: Session to delete.
        """
        await self._ensure_table()

        await self._execute(
            f"DELETE FROM {self._table_name} WHERE session_id = ?",
            [session_id],
        )
