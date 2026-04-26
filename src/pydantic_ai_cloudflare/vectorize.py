"""Cloudflare Vectorize RAG toolset for PydanticAI agents.

Provides search and storage tools backed by Cloudflare Vectorize
(vector database) and Workers AI (embedding models).

Usage::

    from pydantic_ai import Agent
    from pydantic_ai_cloudflare import VectorizeToolset

    agent = Agent(
        "openai:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        toolsets=[VectorizeToolset(index_name="my-docs")],
    )
    result = agent.run_sync("What do our docs say about caching?")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response

logger = logging.getLogger(__name__)

VECTORIZE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5"

_TOOL_DEFS: dict[str, ToolDefinition] = {
    "search_knowledge": ToolDefinition(
        name="search_knowledge",
        description=(
            "Search the knowledge base for information relevant to a query. "
            "Returns the most relevant text passages with similarity scores."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    "store_knowledge": ToolDefinition(
        name="store_knowledge",
        description=(
            "Store a piece of text in the knowledge base for future retrieval. "
            "The text will be embedded and indexed for semantic search."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to store",
                },
                "source": {
                    "type": "string",
                    "description": "Source URL or identifier (optional)",
                    "default": "",
                },
            },
            "required": ["text"],
        },
    ),
}


class VectorizeToolset(AbstractToolset[Any]):
    """Cloudflare Vectorize RAG tools for PydanticAI agents.

    Provides ``search_knowledge`` and ``store_knowledge`` tools backed
    by Cloudflare Vectorize and Workers AI embeddings.

    Args:
        index_name: Vectorize index name.
        embedding_model: Workers AI embedding model ID.
        account_id: Cloudflare account ID.
        api_key: Cloudflare API token.
        top_k: Default number of search results.
        request_timeout: HTTP request timeout in seconds.
        id: Toolset ID.
    """

    def __init__(
        self,
        *,
        index_name: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        account_id: str | None = None,
        api_key: str | None = None,
        top_k: int = 5,
        request_timeout: float = DEFAULT_TIMEOUT,
        id: str | None = "vectorize",
    ) -> None:
        self._index_name = index_name
        self._embedding_model = embedding_model
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._top_k = top_k
        self._timeout = request_timeout
        self._id = id
        self._headers = build_headers(self._api_key)

    @property
    def id(self) -> str | None:
        return self._id

    def _ai_url(self, path: str) -> str:
        """Build Workers AI API URL."""
        return f"{VECTORIZE_BASE_URL}/{self._account_id}/ai/run/{path}"

    def _vectorize_url(self, path: str) -> str:
        """Build Vectorize API URL."""
        return (
            f"{VECTORIZE_BASE_URL}/{self._account_id}"
            f"/vectorize/v2/indexes/{self._index_name}/{path}"
        )

    # -- AbstractToolset interface --

    def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Return the Vectorize tools."""
        return {
            name: ToolsetTool(toolset=self, definition=defn, max_retries=1)
            for name, defn in _TOOL_DEFS.items()
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        """Execute a Vectorize tool."""
        if name == "search_knowledge":
            return await self._search(
                tool_args["query"],
                top_k=tool_args.get("top_k", self._top_k),
            )
        elif name == "store_knowledge":
            return await self._store(
                tool_args["text"],
                source=tool_args.get("source", ""),
            )
        else:
            raise ValueError(f"Unknown Vectorize tool: {name}")

    # -- Tool implementations --

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts using Workers AI.

        # TODO: batch large inputs to avoid hitting the per-request token limit
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._ai_url(self._embedding_model),
                headers=self._headers,
                json={"text": list(texts)},
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        result = data.get("result", {})
        return result.get("data", [])

    async def _search(self, query: str, top_k: int = 5) -> str:
        """Search Vectorize for relevant documents."""
        # Step 1: Embed the query
        embeddings = await self._embed([query])
        if not embeddings:
            return "Failed to generate query embedding."

        query_vector = embeddings[0]

        # Step 2: Query Vectorize
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._vectorize_url("query"),
                headers=self._headers,
                json={
                    "vector": query_vector,
                    "topK": top_k,
                    "returnValues": False,
                    "returnMetadata": "all",
                },
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        matches = data.get("result", {}).get("matches", [])

        if not matches:
            return "No relevant documents found."

        # Format results
        parts: list[str] = []
        for match in matches:
            score = match.get("score", 0.0)
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            source = metadata.get("source", "")
            header = f"[Score: {score:.3f}]"
            if source:
                header += f" Source: {source}"
            parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(parts)

    async def _store(self, text: str, source: str = "") -> str:
        """Embed and store text in Vectorize."""
        import hashlib

        # Step 1: Embed
        embeddings = await self._embed([text])
        if not embeddings:
            return "Failed to generate embedding."

        # Step 2: Generate a deterministic ID
        vector_id = hashlib.sha256(text.encode()).hexdigest()[:32]

        # Step 3: Upsert into Vectorize
        vectors = [
            {
                "id": vector_id,
                "values": embeddings[0],
                "metadata": {
                    "text": text[:1000],  # Vectorize metadata size limit
                    "source": source,
                },
            }
        ]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._vectorize_url("upsert"),
                headers=self._headers,
                json={"vectors": vectors},
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        return f"Stored document (id: {vector_id[:8]}..., {len(text)} chars)."
