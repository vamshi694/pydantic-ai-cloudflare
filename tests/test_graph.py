"""Tests for KnowledgeGraph."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from pydantic_ai_cloudflare.graph import KnowledgeGraph, _cosine_sim, _jaccard

SAMPLE_RECORDS = [
    {
        "id": "A",
        "industry": "SaaS",
        "tech": "AWS, K8s",
        "employees": 1200,
        "desc": "Cloud monitoring platform",
    },
    {
        "id": "B",
        "industry": "SaaS",
        "tech": "AWS, Go",
        "employees": 500,
        "desc": "Cloud storage solution",
    },
    {
        "id": "C",
        "industry": "Security",
        "tech": "Azure, Rust",
        "employees": 300,
        "desc": "Zero trust security",
    },
]


class TestGraphPrimitives:
    def test_cosine_sim(self) -> None:
        assert abs(_cosine_sim([1, 0], [1, 0]) - 1.0) < 0.01
        assert abs(_cosine_sim([1, 0], [0, 1]) - 0.0) < 0.01
        assert _cosine_sim([], []) == 0.0

    def test_jaccard(self) -> None:
        assert _jaccard({1, 2, 3}, {2, 3, 4}) == 2 / 4
        assert _jaccard(set(), set()) == 0.0

    def test_add_node(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            nid = kg._add_node("entity", "NexaTech", {"revenue": "180M"})
            assert nid == "entity:nexatech"
            assert kg._nodes[nid]["label"] == "NexaTech"

    def test_add_edge(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            kg._add_node("entity", "A")
            kg._add_node("industry", "SaaS")
            kg._add_edge("entity:a", "industry:saas", "HAS_INDUSTRY")
            assert len(kg._edges) == 1
            assert "industry:saas" in kg._adj["entity:a"]

    def test_no_duplicate_edges(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            kg._add_node("entity", "A")
            kg._add_node("industry", "SaaS")
            kg._add_edge("entity:a", "industry:saas", "HAS_INDUSTRY")
            kg._add_edge("entity:a", "industry:saas", "HAS_INDUSTRY")
            assert len(kg._edges) == 1


class TestBuildGraph:
    @pytest.mark.asyncio
    async def test_build_basic(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()

            stats = await kg.build_from_records(
                SAMPLE_RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                numeric_columns=["employees"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],  # skip text to avoid API calls
                extract_entities=False,
                compute_similarity=False,
            )

            assert stats["nodes"] > 3  # 3 entities + feature nodes
            assert stats["edges"] > 0
            assert "entity:a" in kg._nodes
            assert "industry:saas" in kg._nodes
            assert "tech:aws" in kg._nodes

    @pytest.mark.asyncio
    async def test_build_with_data_dict(self) -> None:
        from pydantic_ai_cloudflare.data_profiler import profile_data

        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            dd = profile_data(SAMPLE_RECORDS, id_column="id")
            # Override text to skip API calls
            dd.set_type("desc", "categorical")

            kg = KnowledgeGraph()
            stats = await kg.build_from_records(
                SAMPLE_RECORDS,
                data_dict=dd,
                extract_entities=False,
                compute_similarity=False,
            )
            assert stats["nodes"] > 0


class TestFeatures:
    @pytest.mark.asyncio
    async def test_compute_features(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                SAMPLE_RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            features = kg.compute_features()
            assert "A" in features
            assert "degree" in features["A"]
            assert "community_id" in features["A"]
            assert "pagerank" in features["A"]
            assert "clustering_coeff" in features["A"]

    @pytest.mark.asyncio
    async def test_pairwise_features(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                SAMPLE_RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            pf = kg.pairwise_features("A", "B")
            assert "shared_neighbors" in pf
            assert "jaccard" in pf
            assert "adamic_adar" in pf
            assert pf["shared_neighbors"] > 0  # they share SaaS + AWS

    @pytest.mark.asyncio
    async def test_to_feature_dicts(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                SAMPLE_RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            flat = kg.to_feature_dicts()
            assert "A" in flat
            # Flattened edge types should be columns
            assert any("degree" in k for k in flat["A"])


class TestQuery:
    @pytest.mark.asyncio
    async def test_find_similar(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                SAMPLE_RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            similar = await kg.find_similar("A", top_k=2)
            assert len(similar) > 0
            # B should be most similar (shares SaaS + AWS)
            assert similar[0]["entity"] == "B"

    @pytest.mark.asyncio
    async def test_neighborhood(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                SAMPLE_RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            hood = kg.neighborhood("A", hops=1)
            assert len(hood["nodes"]) > 1
            assert len(hood["edges"]) > 0


class TestKNN:
    @pytest.mark.asyncio
    async def test_knn_graph_metric(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                SAMPLE_RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            knn = kg.knn_features(k=2, metric="graph")
            assert "A" in knn
            assert len(knn["A"]["knn_entities"]) <= 2
            assert "knn_avg_distance" in knn["A"]
            assert "knn_min_distance" in knn["A"]
            # B should be nearest to A (shared SaaS + AWS)
            assert knn["A"]["knn_entities"][0] == "B"


class TestKNNRateFeatures:
    @pytest.mark.asyncio
    async def test_knn_rate_features(self) -> None:
        records = [
            {"id": "A", "industry": "SaaS", "tech": "AWS, K8s", "products": "CDN, WAF"},
            {"id": "B", "industry": "SaaS", "tech": "AWS, Go", "products": "CDN, ZT"},
            {"id": "C", "industry": "SaaS", "tech": "AWS, Rust", "products": "WAF, ZT"},
            {"id": "D", "industry": "Security", "tech": "Azure", "products": "ZT"},
        ]
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            rates = kg.knn_rate_features(["products"], k=2, metric="graph")
            assert "A" in rates
            # A's peers (B, C — share SaaS+AWS) should have ZT
            assert "knn_rate_zt" in rates["A"]
            # B and C both have ZT → rate should be 1.0
            assert rates["A"]["knn_rate_zt"] >= 0.5

    @pytest.mark.asyncio
    async def test_recommend(self) -> None:
        records = [
            {"id": "A", "industry": "SaaS", "tech": "AWS", "products": "CDN"},
            {"id": "B", "industry": "SaaS", "tech": "AWS", "products": "CDN, WAF"},
            {"id": "C", "industry": "SaaS", "tech": "AWS", "products": "CDN, WAF, ZT"},
        ]
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            recs = kg.recommend("A", ["products"], k=2, min_rate=0.5)
            # A has CDN. B and C (peers) have WAF. Should recommend WAF.
            values = [r["value"] for r in recs]
            assert "waf" in values

    @pytest.mark.asyncio
    async def test_co_occurrence(self) -> None:
        records = [
            {"id": "A", "products": "CDN, WAF"},
            {"id": "B", "products": "CDN, WAF, ZT"},
            {"id": "C", "products": "CDN"},
            {"id": "D", "products": "WAF, ZT"},
        ]
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                list_columns={"products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )

            co = kg.co_occurrence_features("products")
            # P(WAF|CDN): A,B,C have CDN. A,B have WAF. → 2/3 = 0.667
            assert co["cdn"]["waf"] > 0.5
            # P(CDN|WAF): A,B,D have WAF. A,B,C have CDN. → 2/3 = 0.667
            assert co["waf"]["cdn"] > 0.5


class TestAddRecords:
    @pytest.mark.asyncio
    async def test_incremental_add(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kg = KnowledgeGraph()
            await kg.build_from_records(
                SAMPLE_RECORDS[:2],
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            initial_nodes = len(kg._nodes)

            # Add a new record
            await kg.add_records(
                [SAMPLE_RECORDS[2]],
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            assert len(kg._nodes) > initial_nodes
            assert "entity:c" in kg._nodes
