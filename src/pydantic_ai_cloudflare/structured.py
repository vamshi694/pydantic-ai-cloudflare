"""Smart structured output for Workers AI models.

Workers AI models have quirks that break PydanticAI's default structured
output (tool calling with retries). This module provides:

1. `cf_structured()` — a standalone function that handles schema injection,
   response parsing, and retry logic directly against the Workers AI API,
   bypassing PydanticAI's built-in structured output.

2. Schema utilities for optimizing large schemas.

The approach mirrors what langchain-cloudflare does:
- Inject JSON schema into system prompt
- Set response_format: json_object
- Parse response with fallback strategies (direct, fenced, prose-wrapped)
- Validate with Pydantic
- Retry with error feedback in the prompt (not via API messages)
"""

from __future__ import annotations

import json as json_mod
import logging
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Default Workers AI model
_DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that may contain prose or markdown fencing."""
    # Markdown fenced block
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Find first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            json_mod.loads(candidate)
            return candidate
        except json_mod.JSONDecodeError:
            pass

    return text


def simplify_schema(schema: dict[str, Any], *, keep_descriptions: bool = False) -> dict[str, Any]:
    """Reduce a JSON schema's token footprint while preserving structure."""
    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    strip = {"title", "default", "examples", "example"}
    if not keep_descriptions:
        strip.add("description")

    for key, value in schema.items():
        if key in strip:
            continue
        if key in ("properties", "$defs"):
            result[key] = {
                k: simplify_schema(v, keep_descriptions=keep_descriptions)
                if isinstance(v, dict)
                else v
                for k, v in value.items()
            }
        elif isinstance(value, dict):
            result[key] = simplify_schema(value, keep_descriptions=keep_descriptions)
        elif isinstance(value, list):
            result[key] = [
                simplify_schema(item, keep_descriptions=keep_descriptions)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def schema_stats(model_class: type[BaseModel]) -> dict[str, Any]:
    """Get stats about a Pydantic model's schema complexity."""
    schema = model_class.model_json_schema()
    schema_str = json_mod.dumps(schema)
    defs = schema.get("$defs", {})

    simplified = simplify_schema(schema)
    simplified_str = json_mod.dumps(simplified)

    total_chars = len(schema_str)
    simplified_chars = len(simplified_str)
    field_count = len(schema.get("properties", {}))
    nested_count = len(defs)

    def _max_depth(s: dict, depth: int = 0) -> int:
        max_d = depth
        for v in s.values():
            if isinstance(v, dict):
                max_d = max(max_d, _max_depth(v, depth + 1))
        return max_d

    depth = _max_depth(schema)

    if simplified_chars < 1500:
        rec = "Good -- should work reliably with any Workers AI model."
    elif simplified_chars < 4000:
        rec = "OK -- works with Llama 3.3 70B and Qwen 3 30B. Set max_tokens=4096+."
    elif simplified_chars < 8000:
        rec = (
            "Large -- may need retries. Consider simplify_schema() to reduce size, "
            "or split into multiple agent calls."
        )
    else:
        rec = (
            "Very large -- likely to fail on single call. Split the schema into "
            "sub-schemas and chain multiple agent calls."
        )

    return {
        "total_chars": total_chars,
        "simplified_chars": simplified_chars,
        "reduction": f"{(1 - simplified_chars / total_chars) * 100:.0f}%",
        "field_count": field_count,
        "nested_model_count": nested_count,
        "max_depth": depth,
        "recommendation": rec,
    }


async def cf_structured(
    prompt: str,
    output_type: type[T],
    *,
    model: str = _DEFAULT_MODEL,
    system_prompt: str = "",
    account_id: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    retries: int = 3,
    simplify: bool = True,
) -> T:
    """Get structured output from Workers AI, handling all the quirks.

    This bypasses PydanticAI's structured output entirely and calls the
    Workers AI API directly with schema injection + json_object mode +
    custom retry logic. This is the most reliable way to get complex
    structured output from Workers AI models.

    Works with all Workers AI models: Llama, Qwen, Kimi, Gemma, GLM,
    DeepSeek, Nemotron.

    Args:
        prompt: The user prompt.
        output_type: Pydantic model class for the output.
        model: Workers AI model ID.
        system_prompt: Additional system prompt (prepended to schema instructions).
        account_id: Cloudflare account ID.
        api_key: Cloudflare API token.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature (lower = more deterministic).
        retries: Max retry attempts on validation failure.
        simplify: Whether to simplify the schema to reduce tokens.

    Returns:
        A validated instance of output_type.

    Raises:
        ValidationError: If the model can't produce valid output after retries.
        RuntimeError: If the API returns an error.
    """
    acct = resolve_account_id(account_id)
    token = resolve_api_token(api_key)
    headers = build_headers(token)
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions"

    # Build schema string
    raw_schema = output_type.model_json_schema()
    if simplify:
        display_schema = simplify_schema(raw_schema, keep_descriptions=True)
    else:
        display_schema = raw_schema
    schema_str = json_mod.dumps(display_schema, indent=2)

    # Build system message with schema injection
    schema_instruction = (
        "You must respond with a single valid JSON object that exactly matches "
        "this schema. Do NOT include any text before or after the JSON. "
        "Do NOT wrap in markdown code fences. Do NOT add fields that are not "
        "in the schema. For enum/Literal fields, use EXACTLY the allowed values.\n\n"
        f"JSON Schema:\n{schema_str}"
    )
    full_system = (
        f"{system_prompt}\n\n{schema_instruction}" if system_prompt else schema_instruction
    )

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        # Build messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        # On retries, add the error feedback as a user message
        if attempt > 0 and last_error is not None:
            error_feedback = (
                f"Your previous response had validation errors:\n{last_error}\n\n"
                "Please fix the errors and return valid JSON matching the schema exactly."
            )
            messages.append({"role": "user", "content": error_feedback})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT * 2) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)

        # Extract content
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Handle dict content (Workers AI quirk)
        if isinstance(content, dict):
            content = json_mod.dumps(content)

        if not isinstance(content, str):
            content = str(content)

        # Extract JSON from potential wrapping
        content = extract_json_from_text(content)

        # Try to parse and validate
        try:
            parsed = json_mod.loads(content)
            result = output_type.model_validate(parsed)
            return result
        except (json_mod.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.info(f"Structured output attempt {attempt + 1}/{retries + 1} failed: {e}")
            continue

    # All retries exhausted
    raise last_error or RuntimeError("Failed to get structured output")


def cf_structured_sync(
    prompt: str,
    output_type: type[T],
    **kwargs: Any,
) -> T:
    """Synchronous version of cf_structured().

    Same args as cf_structured() but runs synchronously.
    """
    import asyncio

    return asyncio.run(cf_structured(prompt, output_type, **kwargs))
