"""AI Gateway observability — logs, analytics, feedback.

# FIXME: get_analytics uses the REST endpoint but the GraphQL API
# (aiGatewayRequestsAdaptiveGroups) gives much richer data. Should
# offer both and default to GraphQL when available.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response

logger = logging.getLogger(__name__)

GATEWAY_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"


class GatewayObservability:
    """AI Gateway observability — logs, analytics, cost tracking, feedback.

    AI Gateway automatically logs every LLM request when using
    ``CloudflareProvider``. This class provides programmatic access
    to those logs and analytics.

    Args:
        gateway_id: The AI Gateway ID. Default: ``"default"``.
        account_id: Cloudflare account ID. Falls back to env vars.
        api_key: Cloudflare API token. Falls back to env vars.
        request_timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        *,
        gateway_id: str = "default",
        account_id: str | None = None,
        api_key: str | None = None,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._gateway_id = gateway_id
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._timeout = request_timeout
        self._headers = build_headers(self._api_key)

    def _url(self, path: str = "") -> str:
        """Build the AI Gateway API URL."""
        base = f"{GATEWAY_BASE_URL}/{self._account_id}/ai-gateway/gateways/{self._gateway_id}"
        return f"{base}/{path}" if path else base

    async def get_logs(
        self,
        *,
        limit: int = 20,
        page: int = 1,
        model: str | None = None,
        provider: str | None = None,
        success: bool | None = None,
        cached: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve AI Gateway request logs.

        Args:
            limit: Maximum number of logs to return.
            page: Page number for pagination.
            model: Filter by model name.
            provider: Filter by provider name.
            success: Filter by success status.
            cached: Filter by cache status.

        Returns:
            List of log entries with request/response metadata.
        """
        params: dict[str, Any] = {"per_page": limit, "page": page}
        if model:
            params["model"] = model
        if provider:
            params["provider"] = provider
        if success is not None:
            params["success"] = str(success).lower()
        if cached is not None:
            params["cached"] = str(cached).lower()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url("logs"),
                headers=self._headers,
                params=params,
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        return data.get("result", [])

    async def get_log(self, log_id: str) -> dict[str, Any]:
        """Retrieve a single log entry by ID.

        Args:
            log_id: The log entry ID.

        Returns:
            Log entry with full request/response details.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url(f"logs/{log_id}"),
                headers=self._headers,
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        return data.get("result", {})

    async def add_feedback(
        self,
        log_id: str,
        *,
        score: int | None = None,
        feedback: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add feedback to a log entry.

        Args:
            log_id: The log entry ID.
            score: Numeric score (e.g. 0-100).
            feedback: Thumbs up (1) or down (-1).
            metadata: Additional metadata to attach.
        """
        body: dict[str, Any] = {}
        if score is not None:
            body["score"] = score
        if feedback is not None:
            body["feedback"] = feedback
        if metadata:
            body["metadata"] = metadata

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.patch(
                self._url(f"logs/{log_id}"),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

    async def get_analytics(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Get gateway analytics (request counts, tokens, costs).

        Args:
            start: Start datetime (ISO 8601).
            end: End datetime (ISO 8601).

        Returns:
            Analytics data with totals and breakdowns.
        """
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url("analytics"),
                headers=self._headers,
                params=params,
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        return data.get("result", {})

    async def delete_logs(self) -> None:
        """Delete all logs for this gateway."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.delete(
                self._url("logs"),
                headers=self._headers,
            )
            resp.raise_for_status()
