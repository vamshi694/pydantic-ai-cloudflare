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
        assert profile.supports_tools is True
        assert profile.supports_thinking is True

    def test_qwen_no_json_mode(self) -> None:
        """Qwen is NOT in Workers AI JSON Mode list -- uses tool calling."""
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/qwen/qwen3-30b-a3b")
        assert profile.supports_json_schema_output is False

    def test_gemma_uses_json_object(self) -> None:
        """Gemma uses json_object mode (not tool calling) for structured output."""
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/google/gemma-4-26b-a4b-it")
        assert profile is not None
        assert profile.supports_json_object_output is True
        assert profile.supports_json_schema_output is False
        assert profile.default_structured_output_mode == "json_schema"

    def test_glm_no_tool_choice(self) -> None:
        """GLM doesn't support tool_choice -- can't force tool calling."""
        from pydantic_ai.profiles.openai import OpenAIModelProfile

        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/zai-org/glm-4.7-flash")
        assert profile is not None
        assert isinstance(profile, OpenAIModelProfile)
        assert profile.openai_supports_tool_choice_required is False

    def test_kimi_has_thinking(self) -> None:
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/moonshotai/kimi-k2.6")
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.supports_tools is True

    def test_mistral_no_tool_choice(self) -> None:
        """Mistral doesn't support tool_choice."""
        from pydantic_ai.profiles.openai import OpenAIModelProfile

        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@hf/nousresearch/hermes-2-pro-mistral-7b")
        assert isinstance(profile, OpenAIModelProfile)
        assert profile.openai_supports_tool_choice_required is False

    def test_deepseek_has_json_mode(self) -> None:
        """DeepSeek is in the JSON Mode supported list."""
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/deepseek-ai/deepseek-r1-distill-qwen-32b")
        assert profile.supports_json_schema_output is True
        assert profile.supports_thinking is True

    def test_nemotron_has_thinking(self) -> None:
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/nvidia/nemotron-3-120b-a12b")
        assert profile is not None
        assert profile.supports_thinking is True

    def test_llama_has_json_mode(self) -> None:
        """Llama is in the JSON Mode supported list."""
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        assert profile.supports_json_schema_output is True
        assert profile.supports_json_object_output is True

    def test_all_models_have_schema_transformer(self) -> None:
        """Every model should have OpenAIJsonSchemaTransformer for complex schemas."""

        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        models = [
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/qwen/qwen3-30b-a3b",
            "@cf/google/gemma-4-26b-a4b-it",
            "@cf/zai-org/glm-4.7-flash",
            "@cf/moonshotai/kimi-k2.6",
            "@hf/nousresearch/hermes-2-pro-mistral-7b",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
            "@cf/nvidia/nemotron-3-120b-a12b",
            "@cf/totally/unknown-model",
        ]
        for model in models:
            profile = cloudflare_model_profile(model)
            assert profile is not None, f"No profile for {model}"
            assert profile.json_schema_transformer is not None, f"No transformer for {model}"

    def test_unknown_model_gets_fallback(self) -> None:
        from pydantic_ai_cloudflare.profiles import cloudflare_model_profile

        profile = cloudflare_model_profile("@cf/totally/unknown-model")
        assert profile is not None
        assert profile.supports_tools is True
