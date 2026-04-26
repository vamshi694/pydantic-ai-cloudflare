"""Centralized error types for pydantic-ai-cloudflare."""

from __future__ import annotations


class CloudflareConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


class CloudflareAPIError(RuntimeError):
    """Raised when a Cloudflare API returns an unexpected error."""


# Common error messages
MISSING_ACCOUNT_ID = (
    "Cloudflare account ID is required. Provide it via the account_id parameter "
    "or set the CLOUDFLARE_ACCOUNT_ID environment variable."
)

MISSING_API_TOKEN = (
    "Cloudflare API token is required. Provide it via the api_key parameter "
    "or set the CLOUDFLARE_API_TOKEN environment variable."
)

BROWSER_RUN_BINARY_ERROR = (
    "Browser Run returned {content_type} instead of binary data for "
    "/{mode}. The API may have returned an error."
)
