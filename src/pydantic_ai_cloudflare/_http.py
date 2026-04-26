"""Shared HTTP helpers."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_TIMEOUT = 60.0  # seconds


def build_headers(api_token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def create_async_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


def check_api_response(data: Any) -> None:
    """Raise if CF API returned {success: false}."""
    from ._errors import CloudflareAPIError

    if isinstance(data, dict) and not data.get("success", True):
        errors = data.get("errors", [])
        raise CloudflareAPIError(f"Cloudflare API error: {errors}")
