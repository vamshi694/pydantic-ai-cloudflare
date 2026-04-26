"""Tests for structured output utilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai_cloudflare.structured import (
    extract_json_from_text,
    schema_stats,
    simplify_schema,
)


class SimpleModel(BaseModel):
    name: str
    score: int


class ComplexModel(BaseModel):
    name: str = Field(description="Company name")
    industry: str = Field(default="Unknown", description="Primary industry")
    employees: int = Field(description="Employee count")
    tags: list[str] = Field(default_factory=list, description="Tags")


class NestedModel(BaseModel):
    class Inner(BaseModel):
        value: str = Field(description="The value")
        priority: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"

    title: str = Field(description="Document title with detailed explanation")
    items: list[Inner] = Field(default_factory=list, description="List of items")
    metadata: dict[str, str] | None = None


class TestSimplifySchema:
    def test_strips_descriptions(self) -> None:
        schema = ComplexModel.model_json_schema()
        simplified = simplify_schema(schema)
        # Should not have description in properties
        for prop in simplified.get("properties", {}).values():
            assert "description" not in prop

    def test_keeps_structure(self) -> None:
        schema = ComplexModel.model_json_schema()
        simplified = simplify_schema(schema)
        assert "properties" in simplified
        assert "name" in simplified["properties"]
        assert simplified["properties"]["name"]["type"] == "string"

    def test_keeps_descriptions_when_asked(self) -> None:
        schema = ComplexModel.model_json_schema()
        simplified = simplify_schema(schema, keep_descriptions=True)
        assert "description" in simplified["properties"]["name"]

    def test_reduces_size(self) -> None:
        import json

        schema = NestedModel.model_json_schema()
        original = len(json.dumps(schema))
        simplified = len(json.dumps(simplify_schema(schema)))
        assert simplified < original

    def test_preserves_enum(self) -> None:
        schema = NestedModel.model_json_schema()
        simplified = simplify_schema(schema)
        # The Literal["HIGH", "MEDIUM", "LOW"] should become an enum
        defs = simplified.get("$defs", {})
        inner = defs.get("Inner", {})
        priority_prop = inner.get("properties", {}).get("priority", {})
        # enum should be preserved
        assert "enum" in priority_prop or "anyOf" in priority_prop or "const" in priority_prop


class TestExtractJson:
    def test_fenced_json(self) -> None:
        text = '```json\n{"name": "test"}\n```'
        assert extract_json_from_text(text) == '{"name": "test"}'

    def test_fenced_no_lang(self) -> None:
        text = '```\n{"name": "test"}\n```'
        assert extract_json_from_text(text) == '{"name": "test"}'

    def test_json_with_preamble(self) -> None:
        text = 'Here is the result:\n{"name": "test", "score": 5}'
        result = extract_json_from_text(text)
        assert '"name": "test"' in result

    def test_plain_json(self) -> None:
        text = '{"name": "test"}'
        assert extract_json_from_text(text) == '{"name": "test"}'

    def test_no_json(self) -> None:
        text = "Just some text without JSON"
        assert extract_json_from_text(text) == text


class TestSchemaStats:
    def test_simple_model(self) -> None:
        stats = schema_stats(SimpleModel)
        assert stats["field_count"] == 2
        assert stats["nested_model_count"] == 0
        assert "Good" in stats["recommendation"]

    def test_nested_model(self) -> None:
        stats = schema_stats(NestedModel)
        assert stats["nested_model_count"] >= 1
        assert stats["simplified_chars"] < stats["total_chars"]

    def test_reduction_percentage(self) -> None:
        stats = schema_stats(ComplexModel)
        assert "%" in stats["reduction"]
