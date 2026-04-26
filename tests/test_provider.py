"""Tests for CloudflareProvider."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from pydantic_ai_cloudflare._auth import resolve_account_id, resolve_api_token
from pydantic_ai_cloudflare._errors import CloudflareConfigError
from pydantic_ai_cloudflare.provider import CloudflareProvider


class TestAuthResolution:
    """Verify credential resolution from params and env vars."""

    def test_account_id_from_param(self) -> None:
        assert resolve_account_id("my-id") == "my-id"

    def test_account_id_from_cloudflare_env(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "env-id"}):
            assert resolve_account_id() == "env-id"

    def test_account_id_from_cf_env(self) -> None:
        with patch.dict(os.environ, {"CF_ACCOUNT_ID": "cf-id"}, clear=False):
            # Clear the CLOUDFLARE_ one to test fallback
            env = {k: v for k, v in os.environ.items() if k != "CLOUDFLARE_ACCOUNT_ID"}
            with patch.dict(os.environ, env, clear=True):
                with patch.dict(os.environ, {"CF_ACCOUNT_ID": "cf-id"}):
                    assert resolve_account_id() == "cf-id"

    def test_account_id_missing_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(CloudflareConfigError, match="account ID"):
                resolve_account_id()

    def test_api_token_from_param(self) -> None:
        assert resolve_api_token("my-token") == "my-token"

    def test_api_token_from_cloudflare_env(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_API_TOKEN": "env-token"}, clear=True):
            assert resolve_api_token() == "env-token"

    def test_api_token_from_cf_env(self) -> None:
        with patch.dict(os.environ, {"CF_API_TOKEN": "cf-token"}, clear=True):
            assert resolve_api_token() == "cf-token"

    def test_api_token_from_ai_env(self) -> None:
        with patch.dict(os.environ, {"CF_AI_API_TOKEN": "ai-token"}, clear=True):
            assert resolve_api_token() == "ai-token"

    def test_api_token_missing_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(CloudflareConfigError, match="API token"):
                resolve_api_token()


class TestCloudflareProvider:
    """Verify provider initialization and configuration."""

    def test_name(self) -> None:
        provider = CloudflareProvider(account_id="abc", api_key="tok")
        assert provider.name == "cloudflare"

    def test_base_url_with_gateway(self) -> None:
        provider = CloudflareProvider(account_id="abc123", api_key="tok", gateway_id="default")
        assert provider.base_url == (
            "https://gateway.ai.cloudflare.com/v1/abc123/default/workers-ai/v1"
        )

    def test_base_url_with_custom_gateway(self) -> None:
        provider = CloudflareProvider(account_id="abc123", api_key="tok", gateway_id="production")
        assert "production" in provider.base_url

    def test_base_url_without_gateway(self) -> None:
        provider = CloudflareProvider(account_id="abc123", api_key="tok", gateway_id=None)
        assert provider.base_url == ("https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1")

    def test_client_is_openai(self) -> None:
        from openai import AsyncOpenAI

        provider = CloudflareProvider(account_id="abc", api_key="tok")
        assert isinstance(provider.client, AsyncOpenAI)

    def test_model_profile_llama(self) -> None:
        profile = CloudflareProvider.model_profile("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        assert profile is not None

    def test_model_profile_qwen(self) -> None:
        profile = CloudflareProvider.model_profile("@cf/qwen/qwen3-30b-a3b")
        assert profile is not None

    def test_model_profile_unknown(self) -> None:
        profile = CloudflareProvider.model_profile("@cf/unknown/model")
        # Should return a fallback profile, not None
        assert profile is not None

    def test_gateway_metadata_accepted(self) -> None:
        # Should not raise
        provider = CloudflareProvider(
            account_id="abc",
            api_key="tok",
            gateway_metadata={"session_id": "sess-123", "user_id": 456},
        )
        assert provider._gateway_metadata is not None


class TestModelProfiles:
    """Verify model profiles for Workers AI models."""

    def test_llama_profile(self) -> None:
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        assert profile is not None
        assert profile.supports_tools is True

    def test_qwen3_has_thinking(self) -> None:
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/qwen/qwen3-30b-a3b")
        assert profile is not None

    def test_gemma_profile(self) -> None:
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/google/gemma-4-26b-a4b-it")
        assert profile is not None

    def test_unknown_model_gets_fallback(self) -> None:
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/totally/unknown-model")
        assert profile is not None
