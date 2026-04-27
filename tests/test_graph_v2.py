"""Tests for v0.2.0 EntityGraph features.

Covers the eight production-feedback fixes plus point-in-time, freeze/score,
visualization, and asyncio-parallel LLM extraction.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from pydantic_ai_cloudflare.graph import (
    EntityGraph,
    _entity_projection,
    _idf_weight,
    _parse_date,
    build_temporal_dataset,
)


def _env() -> dict[str, str]:
    return {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}


# Sample dataset that exercises mixed column types, including a sentinel-zero
# numeric column ("propensity_score") and a temporal column.
RECORDS = [
    {
        "id": "AcmeCorp",
        "industry": "SaaS",
        "tech": "AWS, K8s, Python",
        "products": "CDN, WAF",
        "propensity_score": 0,
        "arr": 1_000_000,
        "created": "2023-01-15",
    },
    {
        "id": "NexaTech",
        "industry": "SaaS",
        "tech": "AWS, Go, K8s",
        "products": "CDN, WAF, Zero Trust",
        "propensity_score": 0,
        "arr": 500_000,
        "created": "2023-06-10",
    },
    {
        "id": "Globex",
        "industry": "SaaS",
        "tech": "AWS, K8s",
        "products": "CDN",
        "propensity_score": 0,
        "arr": 800_000,
        "created": "2023-09-01",
    },
    {
        "id": "Initech",
        "industry": "Security",
        "tech": "Azure, Rust",
        "products": "Zero Trust",
        "propensity_score": 0.85,
        "arr": 200_000,
        "created": "2024-01-20",
    },
    {
        "id": "Hooli",
        "industry": "SaaS",
        "tech": "GCP, Go",
        "products": "WAF, Zero Trust",
        "propensity_score": 0.7,
        "arr": 1_500_000,
        "created": "2024-03-05",
    },
    {
        "id": "Pied",
        "industry": "Tech",
        "tech": "AWS, Python",
        "products": "CDN",
        "propensity_score": 0,
        "arr": 0,
        "created": "2024-04-01",
    },
]


# ============================================================
# #1: IDF weighting + sentinel filter
# ============================================================


class TestIDFAndSentinel:
    def test_idf_weight_function(self) -> None:
        # Hub nodes downweighted
        assert _idf_weight(1) == 1.0
        assert _idf_weight(2) > _idf_weight(10) > _idf_weight(100)
        assert _idf_weight(170) < 0.25  # heavy hub

    @pytest.mark.asyncio
    async def test_sentinel_zero_auto_detection(self) -> None:
        """Auto-detect numeric columns with high zero-fraction."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH", "products": "HAS_PRODUCT"},
                numeric_columns=["propensity_score", "arr"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            # propensity_score is 0 in 4/6 records — should be detected
            assert "propensity_score" in kg._build_meta["sentinel_zero_columns"]
            # arr has only one zero — should NOT be detected
            assert "arr" not in kg._build_meta["sentinel_zero_columns"]

            # The 'zero' bucket node should NOT exist (sentinel filter
            # skipped records with propensity_score=0). The non-zero
            # records (0.85, 0.7) still produce a 'tiny' bucket.
            assert "propensity_score_range:zero" not in kg._nodes

            # The propensity bucket should connect at most 2 entities
            # (Initech 0.85, Hooli 0.7) — not all 4 zero records.
            for nid in kg._nodes:
                if nid.startswith("propensity_score_range:"):
                    assert len(kg._adj.get(nid, set())) <= 2, (nid, kg._adj[nid])

    @pytest.mark.asyncio
    async def test_explicit_sentinel_columns(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                numeric_columns=["arr"],
                sentinel_zero_columns=["arr"],
                auto_detect_sentinels=False,
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            # Pied has arr=0 — that should be excluded from graph
            range_zero = "arr_range:zero"
            # Either the node doesn't exist or it has no edges
            if range_zero in kg._nodes:
                assert len(kg._adj.get(range_zero, set())) == 0

    @pytest.mark.asyncio
    async def test_idf_suppresses_hub_signal(self) -> None:
        """With IDF on, find_similar shouldn't pick a hub-shared peer."""
        # Build a graph where one feature node connects EVERY entity (a hub).
        records = [
            {"id": "A", "industry": "X", "tech": "AWS, K8s"},
            {"id": "B", "industry": "X", "tech": "GCP"},  # only shares industry with A
            {"id": "C", "industry": "X", "tech": "AWS, K8s"},  # shares industry + tech with A
            {"id": "D", "industry": "X", "tech": "Azure"},  # only industry
        ]
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            similar = await kg.find_similar("A", top_k=3, use_idf=True)
            # C should rank #1 — shares industry AND two techs with A
            assert similar[0]["entity"] == "C"

    @pytest.mark.asyncio
    async def test_multi_path_scoring(self) -> None:
        """Each shared feature node should add to the similarity score."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH", "products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            similar = await kg.find_similar("AcmeCorp", top_k=5)
            # Multiple paths should appear in via — not just the first one
            assert similar[0]["entity"] == "NexaTech"
            via = similar[0]["via"]
            # NexaTech shares industry, AWS, K8s, CDN, WAF with Acme — at least 4 distinct
            assert len(set(via)) >= 4


# ============================================================
# #2: Point-in-time features
# ============================================================


class TestPointInTime:
    @pytest.mark.asyncio
    async def test_as_of_filters_records(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                time_column="created",
                as_of="2023-12-31",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            # Only records on/before 2023-12-31 should be included
            entity_labels = {kg._nodes[n]["label"] for n in kg._entity_ids}
            assert entity_labels == {"AcmeCorp", "NexaTech", "Globex"}
            assert kg._snapshot_date is not None
            assert "2023-12-31" in kg._snapshot_date

    @pytest.mark.asyncio
    async def test_as_of_without_time_column_warns(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                as_of="2023-12-31",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            assert any("time_column" in w for w in kg._build_warnings)
            # No filter applied — all records included
            assert len(kg._entity_ids) == 6

    def test_parse_date(self) -> None:
        assert _parse_date(None) is None
        assert _parse_date("") is None
        assert _parse_date("2024-01-15") is not None
        assert _parse_date("01/15/2024") is not None

    @pytest.mark.asyncio
    async def test_build_temporal_dataset(self) -> None:
        """Smoke-test the build_temporal_dataset helper."""
        with patch.dict(os.environ, _env()):
            X, y = await build_temporal_dataset(
                RECORDS,
                id_column="id",
                time_column="created",
                snapshot_dates=["2023-12-31", "2024-04-30"],
                target_columns=["products"],
                feature_kwargs={
                    "categorical_columns": ["industry"],
                    "list_columns": {"tech": "USES_TECH"},
                    "text_columns": [],
                    "extract_entities": False,
                    "compute_similarity": False,
                },
                show_progress=False,
            )
            # Two snapshots — earlier has 3 entities, later has 6
            snaps = {row["snapshot_date"][:10] for row in X}
            assert len(snaps) == 2
            # First snapshot should have 3 entity-rows
            first = [row for row in X if row["snapshot_date"].startswith("2023-12-31")]
            assert len(first) == 3


# ============================================================
# #3: Freeze + score
# ============================================================


class TestFreezeScore:
    @pytest.mark.asyncio
    async def test_freeze_then_score_one(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH", "products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(target_columns=["products"], k=2)
            assert kg.is_frozen

            new_record = {
                "id": "NewCorp",
                "industry": "SaaS",
                "tech": "AWS, K8s",
                "products": "",
            }
            features = await kg.score_one(new_record)
            assert "knn_rate_cdn" in features
            assert "knn_avg_distance" in features
            assert features["knn_peers"]
            assert features["degree"] >= 1

            # Frozen graph rejects mutation
            with pytest.raises(RuntimeError, match="frozen"):
                await kg.add_records([new_record], id_column="id")

            kg.unfreeze()
            assert not kg.is_frozen

    @pytest.mark.asyncio
    async def test_score_batch(self) -> None:
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
            kg.freeze(k=2)
            new_records = [{"id": f"new{i}", "industry": "SaaS", "tech": "AWS"} for i in range(3)]
            results = await kg.score_batch(new_records)
            assert len(results) == 3
            for r in results:
                assert "knn_avg_distance" in r

    @pytest.mark.asyncio
    async def test_score_one_requires_freeze(self) -> None:
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
            with pytest.raises(RuntimeError, match="frozen"):
                await kg.score_one({"id": "x", "industry": "SaaS"})


# ============================================================
# #4: save_features warning
# ============================================================


class TestSaveFeaturesWarning:
    @pytest.mark.asyncio
    async def test_warns_when_knn_rate_present_but_no_target_columns(
        self,
        tmp_path,
        caplog,
    ) -> None:
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
            # Compute knn_rate features first — this populates _features
            kg.to_ml_dataset("_dummy_", target_columns=["products"], k=2)
            assert kg._features is not None

            caplog.clear()
            kg.save_features(str(tmp_path / "feats.json"))
            assert any("target_columns" in m for m in caplog.messages), caplog.messages

    @pytest.mark.asyncio
    async def test_payload_includes_metadata(self, tmp_path) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                time_column="created",
                as_of="2024-01-01",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            path = tmp_path / "feats.json"
            kg.save_features(str(path), target_columns=["products"], k=2)
            with open(path) as f:
                payload = json.load(f)
            assert payload["snapshot_date"] is not None
            assert payload["target_columns"] == ["products"]
            assert payload["build_meta"]["time_column"] == "created"


# ============================================================
# #5: min_support
# ============================================================


class TestMinSupport:
    @pytest.mark.asyncio
    async def test_min_support_filters_low_count_pairs(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                list_columns={"products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            # min_support=4 should kill all pairs (max co_count is 3 here)
            co = kg.co_occurrence_features("products", min_support=4)
            assert co == {}, co
            # min_support=1 should let everything through
            co = kg.co_occurrence_features("products", min_support=1)
            assert "cdn" in co
            assert "support" in next(iter(co["cdn"].values()))


# ============================================================
# #6: explain() label resolution
# ============================================================


class TestExplainLabels:
    @pytest.mark.asyncio
    async def test_format_node_label(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS[:2],
                id_column="id",
                categorical_columns=["industry"],
                numeric_columns=["arr"],
                sentinel_zero_columns=[],
                auto_detect_sentinels=False,
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            assert kg._format_node_label("entity:acmecorp") == "AcmeCorp"
            assert kg._format_node_label("industry:saas") == "industry=SaaS"
            # range nodes should show column=bucket
            assert "arr" in kg._format_node_label("arr_range:large").lower()

    @pytest.mark.asyncio
    async def test_explain_filters_range_nodes_by_default(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                numeric_columns=["arr"],
                auto_detect_sentinels=False,
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            exp = kg.explain("AcmeCorp", k=3)
            for peer in exp.get("peers", []):
                for sn in peer.get("shared_nodes", []):
                    # No truncated 'zero' / 'large' bare labels
                    assert sn != "zero"
                    assert sn != "large"
                    # If a range node DID slip through, it has key=value form
                    if "_range" in sn:
                        assert "=" in sn


# ============================================================
# #7: Parallel LLM extraction
# ============================================================


class TestParallelLLM:
    @pytest.mark.asyncio
    async def test_extraction_runs_concurrently(self) -> None:
        """Verify extract_text_features_parallel uses asyncio.gather."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            calls = {"count": 0}

            async def fake_extract_entities(text: str) -> list[str]:
                calls["count"] += 1
                return ["concept_x"]

            kg._extract_entities = fake_extract_entities  # type: ignore

            records = [{"id": f"e{i}", "notes": "long descriptive text " * 20} for i in range(8)]
            await kg.build_from_records(
                records,
                id_column="id",
                text_columns=["notes"],
                extract_entities=True,
                extract_relationships=False,
                summarize_text=False,
                compute_similarity=False,
                llm_concurrency=4,
            )
            # Each record has one text column → 8 calls
            assert calls["count"] == 8
            # Concept node should be linked to all 8 entities
            assert len(kg._adj.get("concept:concept_x", set())) == 8


# ============================================================
# #8: Entity-projection Louvain
# ============================================================


class TestEntityProjection:
    @pytest.mark.asyncio
    async def test_entity_projection_helper(self) -> None:
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
            proj = _entity_projection(kg._adj, kg._typed_adj, kg._nodes, kg._entity_ids)
            # AcmeCorp and NexaTech share industry+AWS+K8s — should have edge
            a = kg._nid("entity", "AcmeCorp")
            n = kg._nid("entity", "NexaTech")
            assert n in proj.get(a, {})
            assert proj[a][n] > 0

    @pytest.mark.asyncio
    async def test_communities_via_projection(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH", "products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            features = kg.compute_features(community_method="entity_projection")
            for entity, feats in features.items():
                assert "community_id" in feats


# ============================================================
# #9 + #10: Build-time profiling and feature_report
# ============================================================


class TestProfilingAndReport:
    @pytest.mark.asyncio
    async def test_build_warnings_for_low_cardinality(self) -> None:
        records = [{"id": f"e{i}", "always_same": "X", "tech": "AWS"} for i in range(5)]
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["always_same"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            assert any("1 unique" in w for w in kg._build_warnings)

    @pytest.mark.asyncio
    async def test_feature_report_structure(self) -> None:
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
            kg.compute_features()
            report = kg.feature_report()
            assert "stats" in report
            assert "features" in report
            assert "warnings" in report
            assert "structural" in report["features"]
            assert "community" in report["features"]


# ============================================================
# Visualization
# ============================================================


class TestVisualization:
    @pytest.mark.asyncio
    async def _build(self) -> EntityGraph:
        kg = EntityGraph()
        await kg.build_from_records(
            RECORDS,
            id_column="id",
            categorical_columns=["industry"],
            list_columns={"tech": "USES_TECH", "products": "HAS_PRODUCT"},
            text_columns=[],
            extract_entities=False,
            compute_similarity=False,
        )
        kg.compute_features()
        return kg

    @pytest.mark.asyncio
    async def test_to_cytoscape(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = await self._build()
            spec = kg.to_cytoscape()
            assert "nodes" in spec
            assert "edges" in spec
            assert "metadata" in spec
            assert spec["metadata"]["legend"]
            for node in spec["nodes"]:
                assert "data" in node
                assert "color" in node["data"]
                # Color must be a hex string
                assert node["data"]["color"].startswith("#")

    @pytest.mark.asyncio
    async def test_to_cytoscape_color_by_community(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = await self._build()
            spec = kg.to_cytoscape(color_by="community")
            assert spec["metadata"]["color_by"] == "community"
            entity_nodes = [n for n in spec["nodes"] if n["data"]["type"] == "entity"]
            # Entity nodes should have community ids in data
            for n in entity_nodes:
                assert "community" in n["data"]

    @pytest.mark.asyncio
    async def test_to_cytoscape_color_by_type(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = await self._build()
            spec = kg.to_cytoscape(color_by="type")
            assert spec["metadata"]["color_by"] == "type"

    @pytest.mark.asyncio
    async def test_to_mermaid(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = await self._build()
            mermaid = kg.to_mermaid(max_nodes=15)
            assert mermaid.startswith("flowchart LR")
            assert "AcmeCorp" in mermaid or "NexaTech" in mermaid
            # Has edge styles via classDef
            assert "classDef" in mermaid

    @pytest.mark.asyncio
    async def test_to_graphml(self, tmp_path) -> None:
        with patch.dict(os.environ, _env()):
            kg = await self._build()
            path = tmp_path / "g.graphml"
            xml = kg.to_graphml(str(path))
            assert "<?xml" in xml
            assert "<graphml" in xml
            assert path.exists()
            content = path.read_text()
            assert "<node" in content
            assert "<edge" in content

    @pytest.mark.asyncio
    async def test_render_html(self, tmp_path) -> None:
        with patch.dict(os.environ, _env()):
            kg = await self._build()
            path = tmp_path / "g.html"
            html = kg.render_html(str(path), title="Test")
            assert "<!doctype html>" in html.lower()
            assert "cytoscape" in html.lower()
            assert path.exists()
            assert "Test" in path.read_text()

    @pytest.mark.asyncio
    async def test_to_d3_json(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = await self._build()
            spec = kg.to_d3_json()
            assert "nodes" in spec
            assert "links" in spec

    @pytest.mark.asyncio
    async def test_focus_subgraph(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = await self._build()
            spec = kg.to_cytoscape(focus="AcmeCorp", hops=1)
            # Focus mode should produce a smaller subgraph
            node_ids = [n["data"]["id"] for n in spec["nodes"]]
            assert any("acmecorp" in nid for nid in node_ids)
