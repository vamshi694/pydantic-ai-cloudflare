"""One-liner agent factory for Cloudflare.

The goal: go from zero to a working agent in one function call.

    from pydantic_ai_cloudflare import cloudflare_agent

    agent = cloudflare_agent()
    result = agent.run_sync("Hello!")

    # With structured output
    agent = cloudflare_agent(output_type=MyModel)

    # With web tools
    agent = cloudflare_agent(web=True)

    # With RAG
    agent = cloudflare_agent(web=True, rag="my-index")

    # Kitchen sink
    agent = cloudflare_agent(
        model="@cf/qwen/qwen3-30b-a3b",
        web=True,
        rag="my-index",
        output_type=MyModel,
        system_prompt="You are a research analyst.",
    )
"""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


def cloudflare_agent(
    model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    *,
    output_type: type[T] | None = None,
    system_prompt: str | None = None,
    web: bool = False,
    web_tools: list[str] | None = None,
    rag: str | None = None,
    retries: int = 3,
    gateway_id: str | None = "default",
    account_id: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a PydanticAI agent wired to Cloudflare in one call.

    This is the fastest way to get a working agent. All Cloudflare
    credentials come from environment variables by default.

    Args:
        model: Workers AI model name. Default: Llama 3.3 70B.
        output_type: Pydantic model for structured output. If None,
            returns plain text.
        system_prompt: System prompt for the agent.
        web: Enable web browsing tools (Browser Run).
        web_tools: Specific web tools to enable. Default: all.
            Options: browse, extract, crawl, scrape, discover_links, screenshot.
        rag: Vectorize index name for RAG. Enables search_knowledge
            and store_knowledge tools.
        retries: Max retries for output validation failures.
        gateway_id: AI Gateway ID. "default" auto-creates. None to skip.
        account_id: Override CLOUDFLARE_ACCOUNT_ID env var.
        api_key: Override CLOUDFLARE_API_TOKEN env var.
        **kwargs: Additional kwargs passed to Agent().

    Returns:
        A configured PydanticAI Agent ready to use.
    """
    from pydantic_ai import Agent

    from .browser_run import BrowserRunToolset
    from .provider import cloudflare_model as _cloudflare_model
    from .vectorize import VectorizeToolset

    cf_model = _cloudflare_model(
        model,
        account_id=account_id,
        api_key=api_key,
        gateway_id=gateway_id,
    )

    toolsets: list[Any] = []
    if web:
        toolsets.append(
            BrowserRunToolset(
                account_id=account_id,
                api_key=api_key,
                tools=web_tools,
            )
        )
    if rag:
        toolsets.append(
            VectorizeToolset(
                index_name=rag,
                account_id=account_id,
                api_key=api_key,
            )
        )

    agent_kwargs: dict[str, Any] = {
        "retries": retries,
        **kwargs,
    }
    if output_type is not None:
        agent_kwargs["output_type"] = output_type
    if system_prompt is not None:
        agent_kwargs["system_prompt"] = system_prompt
    if toolsets:
        agent_kwargs["toolsets"] = toolsets

    # Workers AI default max_tokens is too low for structured output --
    # models truncate JSON mid-field. Set a sensible default unless
    # the user explicitly passed model_settings.
    if "model_settings" not in agent_kwargs:
        from pydantic_ai.settings import ModelSettings

        agent_kwargs["model_settings"] = ModelSettings(max_tokens=4096)

    return Agent(cf_model, **agent_kwargs)
