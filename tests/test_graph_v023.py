"""Tests for v0.2.3 — score_one parity (round 2) + viz edge-case warnings.

v0.2.2's parity fix overshot — score_one emitted MORE keys than to_ml_dataset
because knn_max_distance was sneaking in. The right comparison is against
the UNION of keys across all entities in to_ml_dataset (which is what
pandas.DataFrame.from_dict produces at training time).

This test exercises the proper round-trip:
  X, y = kg.to_ml_dataset(...)            (training side)
  scored = await kg.score_one(record)     (inference side)
  inference_features = {k: v for k, v in scored.items() if isinstance(v, (int, float))}
  set(inference_features) == union of all keys in X.values()
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest

from pydantic_ai_cloudflare import EntityGraph


def _env() -> dict[str, str]:
    return {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}


# Heterogeneous synthetic fixture that triggers per-entity-varying keys —
# entity A has lost_reasons set, entity B doesn't, etc. This forces
# to_ml_dataset's per-entity dicts to differ in size, exposing the
# union-vs-per-entity comparison bug.
RECORDS = [
    # Entity 0: has products + lost_reason + competitor
    {"id": "Acme", "industry": "SaaS", "geo": "NAMER",
     "products": "CDN, WAF", "lost_reason": "price", "competitor": "Zscaler",
     "arr": 1_000_000, "deals_lost": 2},
    # Entity 1: no lost_reason (will lack HAS_LOST_REASON_degree)
    {"id": "Nexa", "industry": "SaaS", "geo": "EMEA",
     "products": "CDN", "lost_reason": "", "competitor": "Okta",
     "arr": 500_000, "deals_lost": 0},
    # Entity 2: no competitor (will lack COMPETES_WITH_degree)
    {"id": "Globex", "industry": "Security", "geo": "APAC",
     "products": "WAF, Zero Trust", "lost_reason": "fit", "competitor": "",
     "arr": 750_000, "deals_lost": 1},
    # Entity 3: nothing on lost_reason or competitor
    {"id": "Initech", "industry": "Security", "geo": "NAMER",
     "products": "Access", "lost_reason": "", "competitor": "",
     "arr": 1_500_000, "deals_lost": 0},
    # Entity 4: complete record again — for variation
    {"id": "Hooli", "industry": "SaaS", "geo": "NAMER",
     "products": "CDN, WAF, Zero Trust", "lost_reason": "timing",
     "competitor": "Cisco", "arr": 2_000_000, "deals_lost": 3},
    # Entity 5: another minimal record
    {"id": "Pied", "industry": "Tech", "geo": "EMEA",
     "products": "CDN", "lost_reason": "", "competitor": "Fortinet",
     "arr": 250_000, "deals_lost": 0},
]


# ============================================================
# Bug #1 (round 2): score_one parity vs the full union schema
# ============================================================


class TestScoreOneParityRound2:
    @pytest.mark.asyncio
    async def test_score_one_keys_subset_of_to_ml_dataset_union(self) -> None:
        """The actual contract: score_one's scalar keys ⊆ training union."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry", "geo"],
                list_columns={"products": "HAS_PRODUCT", "lost_reason": "HAS_LOST_REASON"},
                numeric_columns=["arr", "deals_lost"],
                relationship_columns={"competitor": "COMPETES_WITH"},
                auto_detect_sentinels=False,
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(target_columns=["products"], k=3)

            X, _y = kg.to_ml_dataset("products", target_columns=["products"], k=3)

            # Per-entity dicts have varying key sets — that's expected and
            # is what causes the user's confusion. The TRUE training schema
            # is the union (what pandas.DataFrame.from_dict produces).
            train_union: set[str] = set()
            for feats in X.values():
                train_union.update(feats.keys())

            new_record = {**RECORDS[0], "id": "NEW"}
            scored = await kg.score_one(new_record)

            scalar_keys = {k for k, v in scored.items() if isinstance(v, (int, float))}

            # Equality, not subset — both directions must match.
            assert scalar_keys == train_union, (
                f"score_one scalar keys != to_ml_dataset union\n"
                f"  missing: {sorted(train_union - scalar_keys)}\n"
                f"  extra:   {sorted(scalar_keys - train_union)}"
            )

    @pytest.mark.asyncio
    async def test_score_one_does_not_emit_knn_max_distance(self) -> None:
        """Specific regression: knn_max_distance was the v0.2.2 overshoot."""
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
            scored = await kg.score_one({"id": "X", "industry": "SaaS"})
            # knn_max_distance is NOT in to_ml_dataset's output — must not
            # appear in score_one's output either.
            assert "knn_max_distance" not in scored, (
                "score_one must not emit knn_max_distance — to_ml_dataset doesn't"
            )

    @pytest.mark.asyncio
    async def test_score_one_preserves_knn_peers_for_explainability(self) -> None:
        """knn_peers is a blessed list — preserved for explanation."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(target_columns=["products"], k=3)
            scored = await kg.score_one({"id": "X", "industry": "SaaS",
                                          "products": "CDN"})
            # Top-level list, not nested
            assert "knn_peers" in scored
            assert isinstance(scored["knn_peers"], list)

    @pytest.mark.asyncio
    async def test_inference_dict_feedable_to_model(self) -> None:
        """End-to-end: filter scored dict to scalars → exact match w/ training."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry", "geo"],
                list_columns={"products": "HAS_PRODUCT",
                              "lost_reason": "HAS_LOST_REASON"},
                numeric_columns=["arr"],
                relationship_columns={"competitor": "COMPETES_WITH"},
                auto_detect_sentinels=False,
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(target_columns=["products"], k=3)

            # Training side: simulate pandas DataFrame column extraction
            X, _y = kg.to_ml_dataset("products", target_columns=["products"], k=3)
            train_columns = set()
            for feats in X.values():
                train_columns.update(feats.keys())

            # Inference side: drop non-scalar diagnostics (the documented pattern)
            scored = await kg.score_one({**RECORDS[0], "id": "INFER"})
            inference_features = {k: v for k, v in scored.items()
                                  if isinstance(v, (int, float))}

            assert set(inference_features.keys()) == train_columns

    @pytest.mark.asyncio
    async def test_score_batch_consistent_with_score_one(self) -> None:
        """score_batch(records) and [score_one(r) for r in records] must match."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(target_columns=["products"], k=2)

            new_records = [
                {"id": f"new{i}", "industry": "SaaS", "products": "CDN"}
                for i in range(3)
            ]
            batched = await kg.score_batch(new_records)
            individuals = [await kg.score_one(r) for r in new_records]

            for b, i in zip(batched, individuals):
                # Same scalar feature set
                b_scalars = {k: v for k, v in b.items() if isinstance(v, (int, float))}
                i_scalars = {k: v for k, v in i.items() if isinstance(v, (int, float))}
                assert set(b_scalars.keys()) == set(i_scalars.keys())


# ============================================================
# Bug #2 (round 2): viz edge-case warnings
# ============================================================


class TestVizEdgeCaseWarnings:
    @pytest.mark.asyncio
    async def test_warning_when_filters_remove_everything(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """exclude_node_types covering every type used to silently return empty."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            with caplog.at_level(
                logging.WARNING, logger="pydantic_ai_cloudflare.visualization"
            ):
                spec = kg.to_cytoscape(
                    exclude_node_types=["entity", "industry", "products"],
                )
            # Empty result should be loud, not silent.
            assert spec["nodes"] == []
            assert any(
                "removed all" in m.lower() or "filters" in m.lower()
                for m in caplog.messages
            ), caplog.messages

    @pytest.mark.asyncio
    async def test_focus_resolves_for_full_id_match(self) -> None:
        """Sanity: focus by exact entity ID returns a real subgraph."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"products": "HAS_PRODUCT"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            spec = kg.to_cytoscape(focus="Acme", hops=2)
            assert len(spec["nodes"]) > 0, (
                "to_cytoscape(focus='Acme') must return a non-empty subgraph"
            )
            # And the focus node must be in the result
            ids = {n["data"]["id"] for n in spec["nodes"]}
            assert any("acme" in nid.lower() for nid in ids)
