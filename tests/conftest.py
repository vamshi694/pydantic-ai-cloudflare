"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def account_id() -> str:
    return os.getenv("CLOUDFLARE_ACCOUNT_ID", "test-account-id")


@pytest.fixture(scope="session")
def api_token() -> str:
    return os.getenv("CLOUDFLARE_API_TOKEN", "test-api-token")
