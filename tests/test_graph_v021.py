"""Tests for v0.2.1 EntityGraph UX additions.

Covers the user-experience fixes shipped in v0.2.1:
  - dunder methods (__repr__, __len__, __contains__, __iter__)
  - build_warnings property + automatic logger surfacing
  - quick_build() convenience method
  - GraphConfig dataclass + build_from_config()
  - improved freeze/score error messages
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest

from pydantic_ai_cloudflare import EntityGraph, GraphConfig


def _env() -> dict[str, str]:
    return {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}


# Same fixture data shape as test_graph_v2 but smaller — these tests don't
# need the full corpus to verify UX behavior.
RECORDS = [
    {"id": "AcmeCorp", "industry": "SaaS", "tech": "AWS, K8s", "products": "CDN, WAF"},
    {"id": "NexaTech", "industry": "SaaS", "tech": "AWS, Go, K8s", "products": "CDN, WAF"},
    {"id": "Globex", "industry": "SaaS", "tech": "AWS, K8s", "products": "CDN"},
    {"id": "Initech", "industry": "Security", "tech": "Azure, Rust", "products": "Zero Trust"},
]


# ============================================================
# Dunder methods (__repr__, __len__, __contains__, __iter__)
# ============================================================


class TestDunders:
    def test_repr_empty_graph(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            r = repr(kg)
            assert "EntityGraph" in r
            assert "empty" in r
            assert "default" in r  # default name

    def test_repr_named_graph(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph(name="customers")
            assert "customers" in repr(kg)

    @pytest.mark.asyncio
    async def test_repr_built_graph(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            r = repr(kg)
            # Counts should appear
            assert "entities=4" in r
            assert "nodes=" in r
            assert "edges=" in r
            assert "frozen" not in r  # not frozen

    @pytest.mark.asyncio
    async def test_repr_frozen_graph(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(target_columns=["industry"], k=2)
            assert "frozen" in repr(kg)

    def test_len_empty(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            assert len(kg) == 0

    @pytest.mark.asyncio
    async def test_len_built(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            assert len(kg) == 4

    @pytest.mark.asyncio
    async def test_contains_resolves_entity(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            assert "AcmeCorp" in kg
            # Case-insensitive (resolves via _resolve_entity)
            assert "acmecorp" in kg
            assert "DOES_NOT_EXIST" not in kg
            # Non-string returns False without raising
            assert 42 not in kg
            assert None not in kg

    @pytest.mark.asyncio
    async def test_iter_yields_entity_labels(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            labels = list(kg)
            assert set(labels) == {"AcmeCorp", "NexaTech", "Globex", "Initech"}


# ============================================================
# build_warnings property + automatic logging
# ============================================================


class TestBuildWarnings:
    @pytest.mark.asyncio
    async def test_build_warnings_property_is_readable(self) -> None:
        records = [{"id": f"e{i}", "always_same": "X"} for i in range(5)]
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["always_same"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            ws = kg.build_warnings
            assert isinstance(ws, list)
            # Returns a copy (mutating it doesn't affect the graph)
            ws.append("not stored")
            assert "not stored" not in kg.build_warnings

    @pytest.mark.asyncio
    async def test_build_warnings_logged_at_warning_level(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Profile warnings should now be visible via the logger, not just stored."""
        records = [{"id": f"e{i}", "always_same": "X"} for i in range(5)]
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            with caplog.at_level(logging.WARNING, logger="pydantic_ai_cloudflare.graph"):
                await kg.build_from_records(
                    records,
                    id_column="id",
                    categorical_columns=["always_same"],
                    text_columns=[],
                    extract_entities=False,
                    compute_similarity=False,
                )
            assert any("only 1 unique value" in m for m in caplog.messages), caplog.messages

    @pytest.mark.asyncio
    async def test_as_of_without_time_column_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            with caplog.at_level(logging.WARNING, logger="pydantic_ai_cloudflare.graph"):
                await kg.build_from_records(
                    RECORDS,
                    id_column="id",
                    as_of="2024-01-01",
                    categorical_columns=["industry"],
                    text_columns=[],
                    extract_entities=False,
                    compute_similarity=False,
                )
            assert any("time_column" in m for m in caplog.messages), caplog.messages


# ============================================================
# quick_build() convenience method
# ============================================================


class TestQuickBuild:
    @pytest.mark.asyncio
    async def test_quick_build_basic(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            stats = await kg.quick_build(RECORDS, id_column="id")
            assert stats["nodes"] > 0
            assert stats["edges"] > 0
            # 4 entities should be in the graph
            assert len(kg) == 4
            assert "AcmeCorp" in kg

    @pytest.mark.asyncio
    async def test_quick_build_empty_records(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            stats = await kg.quick_build([])
            assert stats == {"nodes": 0, "edges": 0}
            assert len(kg) == 0

    @pytest.mark.asyncio
    async def test_quick_build_disables_llm_by_default(self) -> None:
        """quick_build should NOT call _extract_entities by default."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            calls = {"count": 0}

            async def fake_extract(text: str) -> list[str]:
                calls["count"] += 1
                return []

            kg._extract_entities = fake_extract  # type: ignore
            await kg.quick_build(RECORDS, id_column="id")
            # use_llm=False → no LLM extraction
            assert calls["count"] == 0

    @pytest.mark.asyncio
    async def test_quick_build_auto_detects_id_column(self) -> None:
        # When id_column is not specified, the profiler should detect "id".
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            stats = await kg.quick_build(RECORDS)  # no id_column
            assert stats["nodes"] > 0
            assert "AcmeCorp" in kg


# ============================================================
# GraphConfig dataclass + build_from_config()
# ============================================================


class TestGraphConfig:
    def test_default_config(self) -> None:
        config = GraphConfig()
        assert config.id_column is None
        assert config.extract_entities is True
        assert config.compute_similarity is True
        assert config.profile is True
        assert config.sentinel_zero_rate == 0.30
        assert config.llm_concurrency == 8

    def test_to_kwargs_round_trip(self) -> None:
        config = GraphConfig(
            id_column="account_id",
            categorical_columns=["industry"],
            extract_entities=False,
        )
        kwargs = config.to_kwargs()
        assert kwargs["id_column"] == "account_id"
        assert kwargs["categorical_columns"] == ["industry"]
        assert kwargs["extract_entities"] is False
        # Unset fields fall back to defaults
        assert kwargs["compute_similarity"] is True

    def test_to_kwargs_passes_through_unknown_objects(self) -> None:
        """data_dict can be a DataDictionary — to_kwargs must not deep-copy it."""

        class FakeDD:
            id_column = "x"

        dd = FakeDD()
        config = GraphConfig(data_dict=dd)
        kwargs = config.to_kwargs()
        # Identity preserved (no deep-copy)
        assert kwargs["data_dict"] is dd

    @pytest.mark.asyncio
    async def test_build_from_config_equivalent(self) -> None:
        """build_from_config should produce the same graph as build_from_records."""
        config = GraphConfig(
            id_column="id",
            categorical_columns=["industry"],
            list_columns={"tech": "USES_TECH"},
            text_columns=[],
            extract_entities=False,
            compute_similarity=False,
        )

        with patch.dict(os.environ, _env()):
            kg_a = EntityGraph()
            stats_a = await kg_a.build_from_config(RECORDS, config)

            kg_b = EntityGraph()
            stats_b = await kg_b.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            assert stats_a == stats_b
            assert len(kg_a) == len(kg_b)
            assert set(kg_a) == set(kg_b)


# ============================================================
# Improved freeze/score error messages
# ============================================================


class TestFreezeErrorMessages:
    @pytest.mark.asyncio
    async def test_score_one_unfrozen_explains_why(self) -> None:
        """Error should explain WHY freeze is needed, not just say 'call freeze'."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            with pytest.raises(RuntimeError) as exc_info:
                await kg.score_one({"id": "x", "industry": "SaaS"})
            msg = str(exc_info.value)
            # Must explain WHY (training/inference parity)
            assert "drift" in msg.lower() or "training" in msg.lower()
            # Must give an alternative for users who don't actually need freeze
            assert "compute_features" in msg or "to_feature_dicts" in msg

    @pytest.mark.asyncio
    async def test_build_on_frozen_graph_explains_options(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(k=2)
            with pytest.raises(RuntimeError) as exc_info:
                await kg.build_from_records(
                    RECORDS,
                    id_column="id",
                    text_columns=[],
                    extract_entities=False,
                    compute_similarity=False,
                )
            msg = str(exc_info.value)
            # Must mention both alternatives — score_one (inference) and unfreeze (rebuild)
            assert "score_one" in msg
            assert "unfreeze" in msg

    @pytest.mark.asyncio
    async def test_add_records_on_frozen_explains_options(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(k=2)
            with pytest.raises(RuntimeError) as exc_info:
                await kg.add_records([{"id": "new", "industry": "SaaS"}], id_column="id")
            msg = str(exc_info.value)
            assert "score_one" in msg
            assert "unfreeze" in msg

    def test_freeze_before_build_gives_useful_error(self) -> None:
        """Calling freeze() before build_from_records() should produce a helpful error."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            kg.freeze(target_columns=["x"], k=2)
            # Now score_one should fail with our metadata-missing error
            import asyncio

            with pytest.raises(RuntimeError) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    kg.score_one({"id": "test"})
                )
            msg = str(exc_info.value)
            # Should reference build_from_records or quick_build
            assert "build_from_records" in msg or "quick_build" in msg
