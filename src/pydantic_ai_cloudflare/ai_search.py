"""Cloudflare AI Search (managed RAG) toolset.

AI Search is Cloudflare's fully-managed RAG pipeline. You upload docs
to R2 or point it at a website, and it handles chunking, embedding,
indexing, and search. This module wraps the search + chat endpoints
as PydanticAI tools.

Requires an AI Search instance created in the dashboard:
  dash.cloudflare.com → AI → AI Search → Create

API token needs: AI Search Edit + AI Search Run permissions.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response

logger = logging.getLogger(__name__)

AI_SEARCH_BASE = "https://api.cloudflare.com/client/v4/accounts"

_TOOL_DEFS: dict[str, ToolDefinition] = {
    "search": ToolDefinition(
        name="search",
        description=(
            "Search the knowledge base for information relevant to a query. "
            "Returns the most relevant text chunks with source attribution."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    ),
    "ask": ToolDefinition(
        name="ask",
        description=(
            "Ask a question and get an AI-generated answer based on the "
            "knowledge base, with source citations."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer"},
            },
            "required": ["question"],
        },
    ),
}


class AISearchToolset(AbstractToolset[Any]):
    """Cloudflare AI Search tools for PydanticAI agents.

    Wraps AI Search's /search and /chat/completions endpoints.
    Requires an AI Search instance created in the Cloudflare dashboard.

    Args:
        instance_name: AI Search instance name (from dashboard).
        account_id: Cloudflare account ID.
        api_key: API token with AI Search Edit + Run permissions.
        tools: Which tools to enable. Default: ["search", "ask"].
        request_timeout: HTTP timeout.
    """

    def __init__(
        self,
        *,
        instance_name: str,
        account_id: str | None = None,
        api_key: str | None = None,
        tools: list[str] | None = None,
        request_timeout: float = DEFAULT_TIMEOUT,
        id: str | None = "ai_search",
    ) -> None:
        self._instance = instance_name
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._enabled = frozenset(tools) if tools else frozenset({"search", "ask"})
        self._timeout = request_timeout
        self._id = id
        self._headers = build_headers(self._api_key)

    @property
    def id(self) -> str | None:
        return self._id

    def _url(self, endpoint: str) -> str:
        return (
            f"{AI_SEARCH_BASE}/{self._account_id}/ai-search/instances/{self._instance}/{endpoint}"
        )

    def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        return {
            name: ToolsetTool(toolset=self, definition=defn, max_retries=1)
            for name, defn in _TOOL_DEFS.items()
            if name in self._enabled
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        if name == "search":
            return await self._search(tool_args["query"])
        elif name == "ask":
            return await self._ask(tool_args["question"])
        raise ValueError(f"Unknown AI Search tool: {name}")

    async def _search(self, query: str) -> str:
        """Search the AI Search instance."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("search"),
                headers=self._headers,
                json={"query": query},
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        result = data.get("result", {})

        # Format chunks
        chunks = result.get("data", result.get("chunks", []))
        if not chunks:
            return "No results found."

        parts = []
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))
            source = chunk.get("filename", chunk.get("source", chunk.get("link", "")))
            score = chunk.get("score", "")
            header = f"[{source}]" if source else ""
            if score:
                header += (
                    f" (score: {score:.2f})" if isinstance(score, float) else f" (score: {score})"
                )
            parts.append(f"{header}\n{text}" if header else text)

        return "\n\n---\n\n".join(parts)

    async def _ask(self, question: str) -> str:
        """Ask AI Search a question (chat completions)."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("chat/completions"),
                headers=self._headers,
                json={
                    "messages": [{"role": "user", "content": question}],
                },
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)

        # OpenAI-compatible response
        choices = data.get("choices", data.get("result", {}).get("choices", []))
        if choices:
            return choices[0].get("message", {}).get("content", "")

        return data.get("result", {}).get("response", str(data))
