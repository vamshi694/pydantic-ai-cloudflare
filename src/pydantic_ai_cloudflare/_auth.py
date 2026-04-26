"""Shared authentication helpers for Cloudflare services."""

from __future__ import annotations

import os

from ._errors import MISSING_ACCOUNT_ID, MISSING_API_TOKEN, CloudflareConfigError


def resolve_account_id(account_id: str | None = None) -> str:
    """Resolve the Cloudflare account ID from parameter or environment.

    Args:
        account_id: Explicit account ID, or None to read from env.

    Returns:
        The resolved account ID.

    Raises:
        CloudflareConfigError: If no account ID is found.
    """
    value = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID") or os.environ.get("CF_ACCOUNT_ID")
    if not value:
        raise CloudflareConfigError(MISSING_ACCOUNT_ID)
    return value


def resolve_api_token(api_key: str | None = None) -> str:
    """Resolve the Cloudflare API token from parameter or environment.

    Args:
        api_key: Explicit API token, or None to read from env.

    Returns:
        The resolved API token.

    Raises:
        CloudflareConfigError: If no API token is found.
    """
    value = (
        api_key
        or os.environ.get("CLOUDFLARE_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
        or os.environ.get("CF_AI_API_TOKEN")
    )
    if not value:
        raise CloudflareConfigError(MISSING_API_TOKEN)
    return value
