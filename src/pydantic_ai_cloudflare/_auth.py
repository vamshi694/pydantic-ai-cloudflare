"""Resolve Cloudflare credentials from params or env vars."""

from __future__ import annotations

import os

from ._errors import MISSING_ACCOUNT_ID, MISSING_API_TOKEN, CloudflareConfigError


def resolve_account_id(account_id: str | None = None) -> str:
    """Check param, then CLOUDFLARE_ACCOUNT_ID, then CF_ACCOUNT_ID."""
    value = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID") or os.environ.get("CF_ACCOUNT_ID")
    if not value:
        raise CloudflareConfigError(MISSING_ACCOUNT_ID)
    return value


def resolve_api_token(api_key: str | None = None) -> str:
    """Check param, then CLOUDFLARE_API_TOKEN, CF_API_TOKEN, CF_AI_API_TOKEN."""
    value = (
        api_key
        or os.environ.get("CLOUDFLARE_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
        or os.environ.get("CF_AI_API_TOKEN")
    )
    if not value:
        raise CloudflareConfigError(MISSING_API_TOKEN)
    return value
