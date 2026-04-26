"""Workers AI model profiles.

Maps model name prefixes to ModelProfile so PydanticAI picks the right
structured output strategy per model.

Key insight from production testing: Workers AI tool calling returns
null arguments on complex schemas (2000+ char JSON schemas, 6+ nested
models). The langchain-cloudflare library solved this by using
json_object mode with schema injection in a system message instead.

We replicate that here: default_structured_output_mode="json_schema"
with supports_json_object_output=True. PydanticAI then uses
response_format={type: "json_object"} and injects the schema into
the system prompt -- avoiding the tool calling bug entirely.

Tool calling is still enabled (supports_tools=True) for regular tool
use (BrowserRunToolset, etc.). It's only structured OUTPUT that
bypasses tool calling in favor of json_object mode.

TODO: Pull model capabilities from the Workers AI API instead of
hardcoding them here.
"""

from __future__ import annotations

from pydantic_ai.models import ModelProfile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

# The schema injection template that gets prepended to the system prompt
# when using json_object mode for structured output. PydanticAI handles
# this via its prompted_output_template when default_structured_output_mode
# is "json_schema" and supports_json_schema_output is False.
_CF_OUTPUT_TEMPLATE = (
    "\nRespond with a single valid JSON object matching this schema. "
    "Do NOT wrap in markdown code fences or backticks. "
    "Do NOT include any text before or after the JSON. "
    "Return ONLY the raw JSON object.\n\n{schema}\n"
)


def cloudflare_model_profile(model_name: str) -> ModelProfile | None:
    """Return an OpenAIModelProfile for a Workers AI model."""
    name = model_name.lower()

    # --- Llama family ---
    # In JSON Mode supported list. Tool calling works for regular tools,
    # but structured output uses json_object mode to avoid the tool
    # calling null-arguments bug on complex schemas.
    if "llama" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,  # avoid tool calling null-args bug
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
        )

    # --- DeepSeek family ---
    # MUST check before Qwen ("deepseek-r1-distill-qwen-32b" contains "qwen").
    # In JSON Mode list. Has reasoning for R1 variants.
    if "deepseek" in name:
        is_r1 = "r1" in name
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
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
    # Tool calling works for tools. Structured output via json_object.
    # Qwen 3 has reasoning_content.
    if "qwen" in name:
        is_qwen3 = "qwen3" in name or "qwen-3" in name
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
            supports_thinking=is_qwen3,
            openai_chat_thinking_field="reasoning_content" if is_qwen3 else None,
            openai_chat_send_back_thinking_parts="field" if is_qwen3 else "auto",
        )

    # --- Kimi family (MoonshotAI K2.6) ---
    # 256K context. Tool calling + reasoning.
    if "kimi" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
            supports_thinking=True,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
        )

    # --- Gemma family ---
    # Tool calling is UNRELIABLE for structured output. Use json_object.
    if "gemma" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
            supports_thinking="gemma-4" in name or "gemma4" in name,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
        )

    # --- GLM family (ZhipuAI GLM-4.7-Flash) ---
    # tool_choice is UNSUPPORTED. json_object mode for structured output.
    if "glm" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=False,
            supports_thinking=True,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
            openai_unsupported_model_settings=(
                "max_tokens",
                "top_k",
                "repetition_penalty",
            ),
        )

    # --- Mistral / Hermes family ---
    # Uses guided_json. tool_choice unsupported.
    if "mistral" in name or "hermes" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=True,
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
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
            supports_thinking=True,
            openai_chat_thinking_field="reasoning_content",
            openai_chat_send_back_thinking_parts="field",
        )

    # --- Llama 4 Scout / Maverick (vision) ---
    if "scout" in name or "maverick" in name:
        return OpenAIModelProfile(
            supports_tools=True,
            supports_json_schema_output=False,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
        )

    # --- Fallback ---
    return OpenAIModelProfile(
        supports_tools=True,
        supports_json_schema_output=False,
        supports_json_object_output=True,
        default_structured_output_mode="tool",
        prompted_output_template=_CF_OUTPUT_TEMPLATE,
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        openai_supports_strict_tool_definition=False,
        openai_supports_tool_choice_required=True,
    )
