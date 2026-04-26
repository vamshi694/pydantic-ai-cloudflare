"""Workers AI model profiles.

Maps model name prefixes to ModelProfile so PydanticAI picks the right
structured output strategy per model.

TODO: Pull model capabilities from the Workers AI API instead of
hardcoding them here. The /models endpoint might expose this eventually.
"""

from __future__ import annotations

from pydantic_ai.models import ModelProfile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile


def cloudflare_model_profile(model_name: str) -> ModelProfile | None:
    """Return an ``OpenAIModelProfile`` for a Workers AI model.

    Args:
        model_name: The full model identifier,
            e.g. ``"@cf/meta/llama-3.3-70b-instruct-fp8-fast"``.

    Returns:
        A profile describing the model's capabilities, or None if unknown.
    """
    name = model_name.lower()

    # --- Llama family ---
    # Full tool calling + structured output + vision (Scout/4.x)
    if "llama" in name:
        profile = OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
        )
        # Llama 4 Scout has vision
        if "scout" in name or "llama-4" in name:
            profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                openai_supports_strict_tool_definition=False,
            )
        return profile

    # --- Qwen family ---
    # Full tool calling + structured output + reasoning (Qwen 3)
    if "qwen" in name:
        profile = OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
        )
        if "qwen3" in name or "qwen-3" in name:
            profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                openai_supports_strict_tool_definition=False,
                openai_chat_thinking_field="reasoning_content",
                openai_chat_send_back_thinking_parts="field",
            )
        return profile

    # --- Gemma family ---
    # Structured output via json_object (not tool calling)
    if "gemma" in name:
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            supports_tools=True,
        )

    # --- Mistral family ---
    if "mistral" in name:
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=False,
        )

    # --- Kimi family ---
    if "kimi" in name:
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
        )

    # --- DeepSeek family ---
    if "deepseek" in name:
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
        )

    # --- Fallback for unknown models ---
    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        openai_supports_strict_tool_definition=False,
    )
