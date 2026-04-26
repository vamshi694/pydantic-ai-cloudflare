"""Workers AI model profiles.

Maps model name prefixes to ModelProfile so PydanticAI picks the right
structured output strategy per model.

The tricky part: each Workers AI model family handles structured output
differently. Some support json_schema via response_format, some only via
tool calling, some need json_object mode with the schema injected into
the system prompt, and Mistral uses guided_json instead of response_format
entirely.

PydanticAI uses ModelProfile to decide HOW to request structured output.
The key fields are:

  supports_tools: can the model do tool calling?
  supports_json_schema_output: can it do response_format={type: "json_schema"}?
  supports_json_object_output: can it do response_format={type: "json_object"}?
  default_structured_output_mode: 'tool' or 'json_schema'
  openai_supports_tool_choice_required: can we force tool calling?
  openai_supports_strict_tool_definition: does it honor "strict": true?

For complex nested schemas (20+ fields, 7+ nested models, Literal types,
Optional, dict, list-of-models), the best strategy depends on the model.

TODO: Pull model capabilities from the Workers AI API instead of
hardcoding them here.
"""

from __future__ import annotations

from pydantic_ai.models import ModelProfile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile


def cloudflare_model_profile(model_name: str) -> ModelProfile | None:
    """Return an OpenAIModelProfile for a Workers AI model."""
    name = model_name.lower()

    # --- Llama family ---
    # Best structured output support on Workers AI.
    # JSON Mode supported (json_schema via response_format).
    # Tool calling works. tool_choice works. Needs tool calls
    # embedded in content for multi-turn conversations.
    # Handles complex nested schemas well (26+ fields, 7+ nested models).
    if "llama" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=True,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
        )

    # --- DeepSeek family ---
    # In the official JSON Mode supported list. Strong tool calling.
    # DeepSeek R1 has reasoning content.
    # MUST check before Qwen because "deepseek-r1-distill-qwen-32b"
    # contains "qwen" in the name.
    if "deepseek" in name:
        is_r1 = "r1" in name
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=True,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
            supports_thinking=is_r1,
            openai_chat_thinking_field="reasoning_content" if is_r1 else None,
            openai_chat_send_back_thinking_parts="field" if is_r1 else "auto",
        )

    # --- Qwen family ---
    # Strong tool calling. NOT in the official JSON Mode supported list,
    # so we use tool calling for structured output (not response_format).
    # Qwen 3 has reasoning_content for chain-of-thought.
    # Handles complex schemas via tool calling.
    if "qwen" in name:
        profile = OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=False,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
        )
        if "qwen3" in name or "qwen-3" in name:
            profile = OpenAIModelProfile(
                supports_tools=True,
                supports_json_schema_output=False,
                supports_json_object_output=False,
                default_structured_output_mode="tool",
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                openai_supports_strict_tool_definition=False,
                openai_supports_tool_choice_required=True,
                supports_thinking=True,
                openai_chat_thinking_field="reasoning_content",
                openai_chat_send_back_thinking_parts="field",
            )
        return profile

    # --- Kimi family (MoonshotAI K2.6) ---
    # 256K context. Strong tool calling + reasoning.
    # NOT in JSON Mode list -- use tool calling for structured output.
    if "kimi" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=False,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
            supports_thinking=True,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
        )

    # --- Gemma family ---
    # Tool calling is UNRELIABLE for structured output (sometimes drops
    # required fields). Use json_object mode instead: set
    # response_format={type: "json_object"} and inject the schema into
    # a system message. PydanticAI handles this when
    # default_structured_output_mode="json_schema" and
    # supports_json_object_output=True but supports_json_schema_output=False.
    if "gemma" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=True,
            default_structured_output_mode="json_schema",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
            # Gemma 4 has reasoning
            supports_thinking="gemma-4" in name or "gemma4" in name,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
        )

    # --- GLM family (ZhipuAI GLM-4.7-Flash) ---
    # Tool calling works, but tool_choice is UNSUPPORTED -- we can't
    # force the model to call our tool. PydanticAI falls back to
    # prompting when tool_choice isn't available.
    # Also: max_tokens, top_k, repetition_penalty are unsupported params.
    if "glm" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=False,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=False,  # key difference
            supports_thinking=True,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
            # GLM doesn't support these model settings
            openai_unsupported_model_settings=(
                "max_tokens",
                "top_k",
                "repetition_penalty",
            ),
        )

    # --- Mistral / Hermes family ---
    # Uses guided_json instead of response_format for structured output.
    # tool_choice is unsupported. Our OpenAIJsonSchemaTransformer still
    # works for the schema transform, but the parameter name is different.
    # PydanticAI's OpenAI provider handles tool calling natively, and the
    # Workers AI API translates response_format → guided_json on their end
    # for the supported Hermes model.
    if "mistral" in name or "hermes" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output="hermes" in name,  # only Hermes is in JSON Mode list
            supports_json_object_output=False,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=False,
        )

    # --- Nemotron family (NVIDIA) ---
    # Tool calling + reasoning.
    if "nemotron" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=False,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
            supports_thinking=True,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
        )

    # --- Llama 4 Scout / Maverick (vision + tool calling) ---
    if "scout" in name or "maverick" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=True,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
        )

    # --- Fallback for unknown models ---
    # Conservative: assume tool calling works but no JSON Mode.
    # strict=false because Workers AI doesn't enforce strict schemas.
    return OpenAIModelProfile(
        supports_tools=True,
        supports_json_schema_output=False,
        supports_json_object_output=False,
        default_structured_output_mode="tool",
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        openai_supports_strict_tool_definition=False,
        openai_supports_tool_choice_required=True,
    )
