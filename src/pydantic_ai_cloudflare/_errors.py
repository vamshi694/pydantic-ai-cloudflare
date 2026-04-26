"""Error types."""

from __future__ import annotations


class CloudflareConfigError(ValueError):
    """Missing or invalid configuration."""


class CloudflareAPIError(RuntimeError):
    """Unexpected API error from Cloudflare."""


MISSING_ACCOUNT_ID = (
    "Cloudflare account ID is required. Set CLOUDFLARE_ACCOUNT_ID env var "
    "or pass account_id parameter."
)

MISSING_API_TOKEN = (
    "Cloudflare API token is required. Set CLOUDFLARE_API_TOKEN env var or pass api_key parameter."
)

BROWSER_RUN_BINARY_ERROR = "Browser Run returned {content_type} instead of binary data for /{mode}."
