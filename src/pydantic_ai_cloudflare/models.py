"""Workers AI model catalog.

Curated list of Workers AI text-generation models with their capabilities.
Useful for discovery and for choosing the right model for your use case.

    from pydantic_ai_cloudflare import list_models, recommend_model

    # See what's available
    for m in list_models():
        print(f"{m['name']}: {m['context']} context, {m['speed']}")

    # Get a recommendation
    model = recommend_model(task="structured_output", schema_size="large")
"""

from __future__ import annotations

from typing import Any

# Curated catalog -- not exhaustive, focused on the models people actually use.
# Updated based on Workers AI docs + real testing.
MODELS: list[dict[str, Any]] = [
    {
        "id": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "name": "Llama 3.3 70B",
        "family": "llama",
        "params": "70B",
        "context": "128K",
        "speed": "fast",
        "tool_calling": True,
        "structured_output": True,
        "reasoning": False,
        "vision": False,
        "best_for": ["general", "structured_output", "tool_calling", "agents"],
    },
    {
        "id": "@cf/meta/llama-3.1-8b-instruct",
        "name": "Llama 3.1 8B",
        "family": "llama",
        "params": "8B",
        "context": "128K",
        "speed": "very_fast",
        "tool_calling": True,
        "structured_output": True,
        "reasoning": False,
        "vision": False,
        "best_for": ["quick_tasks", "simple_schemas", "low_latency"],
    },
    {
        "id": "@cf/qwen/qwen3-30b-a3b",
        "name": "Qwen 3 30B",
        "family": "qwen",
        "params": "30B",
        "context": "128K",
        "speed": "fast",
        "tool_calling": True,
        "structured_output": True,
        "reasoning": True,
        "vision": False,
        "best_for": ["reasoning", "complex_tasks", "structured_output", "agents"],
    },
    {
        "id": "@cf/moonshotai/kimi-k2.6",
        "name": "Kimi K2.6",
        "family": "kimi",
        "params": "MoE",
        "context": "256K",
        "speed": "medium",
        "tool_calling": True,
        "structured_output": True,
        "reasoning": True,
        "vision": False,
        "best_for": ["long_context", "reasoning", "large_documents"],
    },
    {
        "id": "@cf/google/gemma-4-26b-a4b-it",
        "name": "Gemma 4 26B",
        "family": "gemma",
        "params": "26B",
        "context": "128K",
        "speed": "fast",
        "tool_calling": False,  # unreliable
        "structured_output": True,  # via json_object
        "reasoning": True,
        "vision": True,
        "best_for": ["vision", "multimodal", "reasoning"],
    },
    {
        "id": "@cf/zai-org/glm-4.7-flash",
        "name": "GLM 4.7 Flash",
        "family": "glm",
        "params": "unknown",
        "context": "128K",
        "speed": "very_fast",
        "tool_calling": True,
        "structured_output": True,
        "reasoning": True,
        "vision": False,
        "best_for": ["quick_tasks", "reasoning", "low_latency"],
    },
    {
        "id": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "name": "DeepSeek R1 32B",
        "family": "deepseek",
        "params": "32B",
        "context": "128K",
        "speed": "medium",
        "tool_calling": True,
        "structured_output": True,
        "reasoning": True,
        "vision": False,
        "best_for": ["reasoning", "code", "complex_tasks"],
    },
    {
        "id": "@cf/nvidia/nemotron-3-120b-a12b",
        "name": "Nemotron 120B",
        "family": "nemotron",
        "params": "120B MoE",
        "context": "128K",
        "speed": "medium",
        "tool_calling": True,
        "structured_output": True,
        "reasoning": True,
        "vision": False,
        "best_for": ["quality", "reasoning", "complex_tasks"],
    },
    {
        "id": "@cf/meta/llama-3.2-11b-vision-instruct",
        "name": "Llama 3.2 11B Vision",
        "family": "llama",
        "params": "11B",
        "context": "128K",
        "speed": "fast",
        "tool_calling": True,
        "structured_output": True,
        "reasoning": False,
        "vision": True,
        "best_for": ["vision", "image_analysis", "multimodal"],
    },
]


def list_models(
    *,
    family: str | None = None,
    capability: str | None = None,
) -> list[dict[str, Any]]:
    """List available Workers AI models.

    Args:
        family: Filter by model family (llama, qwen, gemma, etc.)
        capability: Filter by capability (tool_calling, reasoning, vision,
            structured_output)

    Returns:
        List of model info dicts.
    """
    results = MODELS
    if family:
        results = [m for m in results if m["family"] == family]
    if capability:
        if capability in ("tool_calling", "reasoning", "vision", "structured_output"):
            results = [m for m in results if m.get(capability)]
        else:
            results = [m for m in results if capability in m.get("best_for", [])]
    return results


def recommend_model(
    *,
    task: str = "general",
    speed: str = "fast",
    schema_size: str = "small",
) -> str:
    """Recommend a Workers AI model for your use case.

    Args:
        task: What you're building. Options: general, reasoning,
            structured_output, vision, agents, code.
        speed: Latency preference. Options: very_fast, fast, medium.
        schema_size: For structured output. Options: small (< 5 fields),
            medium (5-15 fields), large (15+ fields).

    Returns:
        A model ID string ready to pass to cloudflare_model().
    """
    if task == "vision":
        return "@cf/google/gemma-4-26b-a4b-it"

    if task == "code":
        return "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"

    if task == "reasoning":
        if speed == "very_fast":
            return "@cf/zai-org/glm-4.7-flash"
        return "@cf/qwen/qwen3-30b-a3b"

    if schema_size == "large":
        # Largest context + best instruction following
        return "@cf/moonshotai/kimi-k2.6"

    if speed == "very_fast":
        return "@cf/meta/llama-3.1-8b-instruct"

    # Default: best all-rounder
    return "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
