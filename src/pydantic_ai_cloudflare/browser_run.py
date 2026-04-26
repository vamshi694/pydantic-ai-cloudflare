"""Browser Run toolset for PydanticAI agents.

Gives agents the ability to browse the web, extract structured data,
crawl sites, scrape elements, discover links, and take screenshots --
all via Cloudflare Browser Run's REST API.

Usage::

    from pydantic_ai import Agent
    from pydantic_ai_cloudflare import BrowserRunToolset

    agent = Agent(
        "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        toolsets=[BrowserRunToolset()],
    )
    result = agent.run_sync("What's on the Cloudflare pricing page?")
"""

from __future__ import annotations

import base64
import json as json_mod
import logging
import time
import warnings
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from ._auth import resolve_account_id, resolve_api_token
from ._errors import CloudflareAPIError
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response

logger = logging.getLogger(__name__)

BROWSER_RUN_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"

# All available tool names
ALL_TOOLS = frozenset({"browse", "extract", "crawl", "scrape", "discover_links", "screenshot"})

# Tool definitions for the LLM
_TOOL_DEFS: dict[str, ToolDefinition] = {
    "browse": ToolDefinition(
        name="browse",
        description=(
            "Fetch a web page and return its content as clean markdown. "
            "Handles JavaScript-rendered pages. Input: a URL string."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch"}},
            "required": ["url"],
        },
    ),
    "extract": ToolDefinition(
        name="extract",
        description=(
            "Extract structured data from a web page using AI. "
            "Provide a URL and a natural language prompt describing what to extract. "
            "Returns structured JSON data."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to extract data from"},
                "prompt": {
                    "type": "string",
                    "description": "What to extract, e.g. 'Extract pricing plans'",
                },
            },
            "required": ["url", "prompt"],
        },
    ),
    "crawl": ToolDefinition(
        name="crawl",
        description=(
            "Crawl a website starting from a URL, following links up to a "
            "configurable depth and page limit. Returns markdown content from "
            "each discovered page. Useful for indexing entire documentation sites."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The seed URL to start crawling from"},
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to crawl (default: 10)",
                    "default": 10,
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum link depth from seed URL (default: 2)",
                    "default": 2,
                },
            },
            "required": ["url"],
        },
    ),
    "scrape": ToolDefinition(
        name="scrape",
        description=(
            "Extract specific elements from a web page using CSS selectors. "
            "Returns the text content of matched elements."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to scrape"},
                "selectors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CSS selectors to match, e.g. ['h1', '.price', 'nav a']",
                },
            },
            "required": ["url", "selectors"],
        },
    ),
    "discover_links": ToolDefinition(
        name="discover_links",
        description=(
            "Discover all links on a web page. Returns a list of URLs found. "
            "Useful for research agents that need to explore a site."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to scan for links"}},
            "required": ["url"],
        },
    ),
    "screenshot": ToolDefinition(
        name="screenshot",
        description=("Capture a screenshot of a web page. Returns a base64-encoded PNG image."),
        parameters_json_schema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to screenshot"}},
            "required": ["url"],
        },
    ),
}


class BrowserRunToolset(AbstractToolset[Any]):
    """Cloudflare Browser Run tools for PydanticAI agents.

    Provides web interaction tools backed by
    `Browser Run <https://developers.cloudflare.com/browser-run/>`_
    Quick Actions REST API.

    Args:
        account_id: Cloudflare account ID. Falls back to env vars.
        api_key: Cloudflare API token with Browser Rendering - Edit permission.
        tools: Subset of tools to expose. Default: all tools.
            Options: ``"browse"``, ``"extract"``, ``"crawl"``, ``"scrape"``,
            ``"discover_links"``, ``"screenshot"``.
        request_timeout: Timeout in seconds for HTTP requests.
        crawl_timeout: Maximum seconds to wait for a crawl job.
        crawl_poll_interval: Seconds between crawl status polls.
        id: Toolset ID for disambiguation when using multiple toolsets.
    """

    def __init__(
        self,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        tools: Sequence[str] | None = None,
        request_timeout: float = DEFAULT_TIMEOUT,
        crawl_timeout: float = 300.0,
        crawl_poll_interval: float = 2.0,
        id: str | None = "browser_run",
    ) -> None:
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._enabled_tools = frozenset(tools) if tools else ALL_TOOLS
        self._request_timeout = request_timeout
        self._crawl_timeout = crawl_timeout
        self._crawl_poll_interval = crawl_poll_interval
        self._id = id
        self._headers = build_headers(self._api_key)

    @property
    def id(self) -> str | None:
        return self._id

    def _url(self, endpoint: str) -> str:
        """Build the Browser Run API URL for an endpoint."""
        return f"{BROWSER_RUN_BASE_URL}/{self._account_id}/browser-rendering/{endpoint}"

    # -- AbstractToolset interface --

    def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Return the enabled Browser Run tools."""
        tools: dict[str, ToolsetTool[Any]] = {}
        for name in self._enabled_tools:
            if name in _TOOL_DEFS:
                tools[name] = ToolsetTool(
                    toolset=self,
                    definition=_TOOL_DEFS[name],
                    max_retries=1,
                )
        return tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        """Execute a Browser Run tool."""
        if name == "browse":
            return await self._browse(tool_args["url"])
        elif name == "extract":
            return await self._extract(tool_args["url"], tool_args["prompt"])
        elif name == "crawl":
            return await self._crawl(
                tool_args["url"],
                max_pages=tool_args.get("max_pages", 10),
                depth=tool_args.get("depth", 2),
            )
        elif name == "scrape":
            return await self._scrape(tool_args["url"], tool_args["selectors"])
        elif name == "discover_links":
            return await self._discover_links(tool_args["url"])
        elif name == "screenshot":
            return await self._screenshot(tool_args["url"])
        else:
            raise ValueError(f"Unknown Browser Run tool: {name}")

    # -- Tool implementations --

    async def _browse(self, url: str) -> str:
        """Fetch a page as markdown via /markdown endpoint."""
        async with httpx.AsyncClient(timeout=self._request_timeout) as client:
            resp = await client.post(
                self._url("markdown"),
                headers=self._headers,
                json={"url": url},
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        return str(data.get("result", ""))

    async def _extract(self, url: str, prompt: str) -> str:
        """Extract structured data via /json endpoint."""
        async with httpx.AsyncClient(timeout=self._request_timeout) as client:
            resp = await client.post(
                self._url("json"),
                headers=self._headers,
                json={"url": url, "prompt": prompt},
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        result = data.get("result", {})
        return json_mod.dumps(result, indent=2) if isinstance(result, dict) else str(result)

    async def _crawl(self, url: str, max_pages: int = 10, depth: int = 2) -> str:
        """Crawl a site via /crawl endpoint (async polling)."""
        async with httpx.AsyncClient(timeout=self._request_timeout) as client:
            # Start the crawl job
            resp = await client.post(
                self._url("crawl"),
                headers=self._headers,
                json={
                    "url": url,
                    "limit": max_pages,
                    "depth": depth,
                    "formats": ["markdown"],
                },
            )
            resp.raise_for_status()
            job_id = resp.json().get("result", "")

            if not job_id:
                return "Crawl job failed to start."

            # Poll for completion
            import asyncio

            results_url = f"{self._url('crawl')}/{job_id}"
            start_time = time.time()

            while True:
                elapsed = time.time() - start_time
                if elapsed > self._crawl_timeout:
                    warnings.warn(
                        f"Crawl for {url} timed out after {self._crawl_timeout}s. "
                        "Returning partial results.",
                        stacklevel=2,
                    )
                    break

                poll = await client.get(results_url, headers=self._headers)
                poll.raise_for_status()
                poll_data = poll.json().get("result", {})
                status = poll_data.get("status", "")

                if status in (
                    "completed",
                    "errored",
                    "cancelled_by_user",
                    "cancelled_due_to_timeout",
                    "cancelled_due_to_limits",
                ):
                    break

                await asyncio.sleep(self._crawl_poll_interval)

            # Collect results
            pages: list[str] = []
            poll_final = await client.get(results_url, headers=self._headers)
            poll_final.raise_for_status()
            final_data = poll_final.json().get("result", {})

            for record in final_data.get("records", []):
                if record.get("status") == "completed":
                    content = record.get("markdown", "")
                    page_url = record.get("url", "")
                    if content:
                        pages.append(f"## {page_url}\n\n{content[:2000]}")

        if not pages:
            return "No pages were crawled successfully."
        return "\n\n---\n\n".join(pages)

    async def _scrape(self, url: str, selectors: list[str]) -> str:
        """Scrape elements via /scrape endpoint."""
        elements = [{"selector": s} for s in selectors]
        async with httpx.AsyncClient(timeout=self._request_timeout) as client:
            resp = await client.post(
                self._url("scrape"),
                headers=self._headers,
                json={"url": url, "elements": elements},
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)

        parts: list[str] = []
        for group in data.get("result", []):
            selector = group.get("selector", "")
            texts = [r.get("text", "") for r in group.get("results", [])]
            combined = "\n".join(t for t in texts if t)
            if combined:
                parts.append(f"[{selector}]\n{combined}")

        return "\n\n".join(parts) if parts else "No matching elements found."

    async def _discover_links(self, url: str) -> str:
        """Discover links via /links endpoint."""
        async with httpx.AsyncClient(timeout=self._request_timeout) as client:
            resp = await client.post(
                self._url("links"),
                headers=self._headers,
                json={"url": url},
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        links = data.get("result", [])
        return "\n".join(links) if links else "No links found."

    async def _screenshot(self, url: str) -> str:
        """Capture a screenshot via /screenshot endpoint."""
        async with httpx.AsyncClient(timeout=self._request_timeout) as client:
            resp = await client.post(
                self._url("screenshot"),
                headers=self._headers,
                json={"url": url},
            )
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type or "text/html" in content_type:
            data = resp.json()
            check_api_response(data)
            raise CloudflareAPIError(
                f"Browser Run returned {content_type} instead of binary data "
                f"for /screenshot: {data}"
            )

        return base64.b64encode(resp.content).decode("utf-8")
