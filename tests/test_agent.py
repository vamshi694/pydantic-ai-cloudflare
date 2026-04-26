"""Tests for cloudflare_agent() factory."""

from __future__ import annotations

import os
from unittest.mock import patch

from pydantic import BaseModel

from pydantic_ai_cloudflare.agent import cloudflare_agent


class TestAgentFactory:
    def test_creates_basic_agent(self) -> None:
        with patch.dict(
            os.environ,
            {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "tok"},
        ):
            agent = cloudflare_agent()
            assert agent is not None

    def test_with_output_type(self) -> None:
        class MyOutput(BaseModel):
            name: str

        with patch.dict(
            os.environ,
            {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "tok"},
        ):
            agent = cloudflare_agent(output_type=MyOutput)
            assert agent._output_type is not None

    def test_with_system_prompt(self) -> None:
        with patch.dict(
            os.environ,
            {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "tok"},
        ):
            agent = cloudflare_agent(system_prompt="Be concise")
            assert agent is not None

    def test_with_web_tools(self) -> None:
        with patch.dict(
            os.environ,
            {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "tok"},
        ):
            agent = cloudflare_agent(web=True)
            assert agent is not None

    def test_with_specific_web_tools(self) -> None:
        with patch.dict(
            os.environ,
            {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "tok"},
        ):
            agent = cloudflare_agent(web=True, web_tools=["browse", "extract"])
            assert agent is not None

    def test_with_rag(self) -> None:
        with patch.dict(
            os.environ,
            {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "tok"},
        ):
            agent = cloudflare_agent(rag="my-index")
            assert agent is not None

    def test_custom_model(self) -> None:
        with patch.dict(
            os.environ,
            {"CLOUDFLARE_ACCOUNT_ID": "test", "CLOUDFLARE_API_TOKEN": "tok"},
        ):
            agent = cloudflare_agent(model="@cf/qwen/qwen3-30b-a3b")
            assert agent is not None


class TestModels:
    def test_list_models(self) -> None:
        from pydantic_ai_cloudflare.models import list_models

        models = list_models()
        assert len(models) > 5
        names = [m["name"] for m in models]
        assert "Llama 3.3 70B" in names

    def test_list_models_by_family(self) -> None:
        from pydantic_ai_cloudflare.models import list_models

        llama = list_models(family="llama")
        assert all(m["family"] == "llama" for m in llama)

    def test_list_models_by_capability(self) -> None:
        from pydantic_ai_cloudflare.models import list_models

        reasoning = list_models(capability="reasoning")
        assert all(m["reasoning"] for m in reasoning)

    def test_recommend_model(self) -> None:
        from pydantic_ai_cloudflare.models import recommend_model

        assert "llama-3.3-70b" in recommend_model()
        assert "gemma" in recommend_model(task="vision")
        assert "deepseek" in recommend_model(task="code")
        assert "kimi" in recommend_model(schema_size="large")
        assert "8b" in recommend_model(speed="very_fast")
