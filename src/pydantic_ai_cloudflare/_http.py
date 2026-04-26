"""Shared HTTP client helpers with timeout enforcement."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_TIMEOUT = 60.0  # seconds


def build_headers(api_token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build standard authorization headers.

    Args:
        api_token: The Cloudflare API token.
        extra: Additional headers to merge.

    Returns:
        Headers dict with Authorization and Content-Type.
    """
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def create_async_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    """Create an httpx async client with the configured timeout.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        A configured httpx.AsyncClient.
    """
    return httpx.AsyncClient(timeout=timeout)


def check_api_response(data: Any) -> None:
    """Raise if the Cloudflare API returned a ``success=false`` envelope.

    Args:
        data: Parsed JSON response body.

    Raises:
        CloudflareAPIError: When the API indicates failure.
    """
    from ._errors import CloudflareAPIError

    if isinstance(data, dict) and not data.get("success", True):
        errors = data.get("errors", [])
        raise CloudflareAPIError(f"Cloudflare API error: {errors}")
