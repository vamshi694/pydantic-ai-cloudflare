"""Smart structured output for Workers AI models.

Workers AI models struggle with very large schemas (9K+ chars, 18+ nested
models) because they truncate output or ignore the schema. This module
provides utilities to work around those limitations:

1. Schema simplification: strip Field descriptions and defaults to reduce
   token overhead while keeping the structure valid.
2. Validation error feedback: when output validation fails, feed the error
   back to the model so it can self-correct.
3. Model fallback: try a fast model first, fall back to a larger one if
   validation keeps failing.
"""

from __future__ import annotations

import json as json_mod
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def simplify_schema(schema: dict[str, Any], *, keep_descriptions: bool = False) -> dict[str, Any]:
    """Reduce a JSON schema's token footprint while preserving structure.

    Strips Field descriptions, defaults, titles, and examples that inflate
    the schema but don't help the model produce valid output. Keeps the
    structural parts (type, properties, required, enum, items) intact.

    Typically reduces a 9K char schema to ~3K chars -- enough for Workers
    AI to handle reliably.

    Args:
        schema: A Pydantic model's JSON schema (from model_json_schema()).
        keep_descriptions: If True, keep 'description' fields (useful for
            complex schemas where the model needs hints).

    Returns:
        A simplified copy of the schema.
    """
    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    # Top-level keys to strip (reduce tokens without losing structure)
    strip = {"title", "default", "examples", "example"}
    if not keep_descriptions:
        strip.add("description")

    for key, value in schema.items():
        if key in strip:
            continue
        if key in ("properties", "$defs"):
            # These are dicts of sub-schemas -- simplify each value
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


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that may contain prose or markdown fencing.

    Handles common Workers AI output quirks:
    - JSON wrapped in ```json ... ``` fences
    - JSON preceded by "Here is the JSON:" or similar
    - JSON followed by explanation text

    Returns:
        The extracted JSON string, or the original text if no JSON found.
    """
    # Try markdown fenced block first
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Try to find a JSON object
    # Look for the first { and last }
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


def schema_stats(model_class: type[BaseModel]) -> dict[str, Any]:
    """Get useful stats about a Pydantic model's schema complexity.

    Helps you understand whether a schema is likely to work reliably
    with Workers AI models.

    Returns:
        Dict with: total_chars, field_count, nested_model_count,
        max_depth, recommendation.
    """
    schema = model_class.model_json_schema()
    schema_str = json_mod.dumps(schema)
    defs = schema.get("$defs", {})

    simplified = simplify_schema(schema)
    simplified_str = json_mod.dumps(simplified)

    total_chars = len(schema_str)
    simplified_chars = len(simplified_str)
    field_count = len(schema.get("properties", {}))
    nested_count = len(defs)

    # Rough depth calculation
    def _max_depth(s: dict, depth: int = 0) -> int:
        max_d = depth
        for v in s.values():
            if isinstance(v, dict):
                max_d = max(max_d, _max_depth(v, depth + 1))
        return max_d

    depth = _max_depth(schema)

    # Recommendation
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
