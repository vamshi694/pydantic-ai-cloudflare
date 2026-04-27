"""Tests for v0.2.2 EntityGraph bug fixes.

Each test class targets one of the 8 fixes shipped in v0.2.2 from the CF1
field-test feedback. Numbered to match the CHANGELOG sections.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest

import pydantic_ai_cloudflare.graph as graph_mod
from pydantic_ai_cloudflare import EntityGraph


def _env() -> dict[str, str]:
    return {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}


def _env_without_creds() -> dict[str, str]:
    """A copy of os.environ with all Cloudflare-related vars removed."""
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CF") and not k.startswith("CLOUDFLARE")
    }


# Fixture: heterogeneous CF1-style records — categorical + list + numeric
# + relationship column, mixed null patterns, time column.
RECORDS = [
    {
        "id": f"A{i}",
        "industry": "SaaS" if i % 2 == 0 else "Security",
        "tech": "AWS, K8s" if i % 2 == 0 else "Azure, Rust",
        "products": "CDN, WAF" if i % 3 == 0 else "Zero Trust, Browser Isolation",
        "arr": 1_000_000 + i * 100_000,
        "competitor": "Zscaler" if i % 2 == 0 else "Okta",
        "created": f"2023-0{(i % 9) + 1}-15",
    }
    for i in range(8)
]


# ============================================================
# Fix #1: score_one() feature parity with to_ml_dataset()
# ============================================================


class TestScoreOneFeatureParity:
    @pytest.mark.asyncio
    async def test_score_one_matches_to_ml_dataset_keys(self) -> None:
        """score_one(record) must emit every key that to_ml_dataset() emits."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH", "products": "HAS_PRODUCT"},
                numeric_columns=["arr"],
                relationship_columns={"competitor": "COMPETES_WITH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(target_columns=["products"], k=3)

            X, _y = kg.to_ml_dataset("products", target_columns=["products"], k=3)
            train_keys = set(next(iter(X.values())).keys())

            new_record = {
                "id": "NEW",
                "industry": "SaaS",
                "tech": "AWS, K8s",
                "products": "",
                "arr": 500_000,
                "competitor": "Zscaler",
            }
            scored = await kg.score_one(new_record)
            score_keys = set(scored.keys())

            missing = train_keys - score_keys
            assert missing == set(), (
                f"score_one() missing {len(missing)} feature keys "
                f"that to_ml_dataset() produces: {sorted(missing)}"
            )

    @pytest.mark.asyncio
    async def test_score_one_includes_numeric_range_degree(self) -> None:
        """v0.2.1 only emitted *_degree for categorical/list. Now numeric too."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                numeric_columns=["arr"],
                auto_detect_sentinels=False,
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(k=2)
            scored = await kg.score_one({"id": "X", "arr": 750_000})
            assert "IN_ARR_RANGE_degree" in scored
            assert scored["IN_ARR_RANGE_degree"] == 1

    @pytest.mark.asyncio
    async def test_score_one_includes_relationship_degree(self) -> None:
        """relationship_columns produce {rel_type}_degree features now."""
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                relationship_columns={"competitor": "COMPETES_WITH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.freeze(k=2)
            scored = await kg.score_one(
                {"id": "X", "competitor": "Zscaler"}
            )
            assert "COMPETES_WITH_degree" in scored
            assert scored["COMPETES_WITH_degree"] == 1

    @pytest.mark.asyncio
    async def test_score_one_fills_missing_keys_with_zero(self) -> None:
        """Schema parity: keys not applicable to a record default to 0."""
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
            # Record missing the 'tech' field entirely
            scored = await kg.score_one({"id": "X", "industry": "SaaS"})
            # USES_TECH_degree should still be present (0)
            assert "USES_TECH_degree" in scored
            assert scored["USES_TECH_degree"] == 0


# ============================================================
# Fix #2: build_temporal_dataset() works without CF creds
# ============================================================


class TestNoCredentialsRequired:
    def test_entity_graph_instantiable_without_creds(self) -> None:
        """EntityGraph() must NOT raise when CLOUDFLARE_* env vars are unset."""
        env_clean = _env_without_creds()
        with patch.dict(os.environ, env_clean, clear=True):
            kg = EntityGraph()
            assert kg._account_id is None
            assert kg._api_key is None

    @pytest.mark.asyncio
    async def test_build_from_records_works_without_creds(self) -> None:
        """Pure-structural build (extract_entities=False) needs no creds."""
        env_clean = _env_without_creds()
        with patch.dict(os.environ, env_clean, clear=True):
            kg = EntityGraph()
            stats = await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            assert stats["nodes"] > 0
            assert len(kg) == 8

    @pytest.mark.asyncio
    async def test_build_temporal_dataset_works_without_creds(self) -> None:
        """The user-reported showstopper: build_temporal_dataset on a CSV."""
        from pydantic_ai_cloudflare import build_temporal_dataset

        env_clean = _env_without_creds()
        with patch.dict(os.environ, env_clean, clear=True):
            X, y = await build_temporal_dataset(
                RECORDS,
                id_column="id",
                time_column="created",
                snapshot_dates=["2023-04-30", "2023-09-30"],
                target_columns=["products"],
                feature_kwargs={
                    "list_columns": {"tech": "USES_TECH", "products": "HAS_PRODUCT"},
                    "text_columns": [],
                    "extract_entities": False,
                    "compute_similarity": False,
                },
                show_progress=False,
            )
            # Two snapshots, multiple entities each
            assert len(X) > 0
            assert len(y) == len(X)

    @pytest.mark.asyncio
    async def test_real_api_call_still_raises_clear_error(self) -> None:
        """Lazy creds shouldn't hide auth failures from real API calls."""
        env_clean = _env_without_creds()
        with patch.dict(os.environ, env_clean, clear=True):
            kg = EntityGraph()
            from pydantic_ai_cloudflare._errors import CloudflareConfigError

            with pytest.raises(CloudflareConfigError, match="account ID"):
                await kg._embed(["hello"])


# ============================================================
# Fix #3: gensim warning fires once total per process
# ============================================================


class TestGensimWarningOnce:
    def test_node2vec_helper_warns_once(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Even if _node2vec_embeddings is called many times, warn only once."""
        # Reset the module-level flag so this test is deterministic.
        graph_mod._GENSIM_WARNED = False

        with patch.dict(os.environ, _env()):
            # Try to import gensim — if it's actually installed, skip
            try:
                import gensim  # noqa: F401

                pytest.skip("gensim is installed in this environment")
            except ImportError:
                pass

            # Force the warning path
            with caplog.at_level(
                logging.WARNING, logger="pydantic_ai_cloudflare.graph"
            ):
                # Three separate calls — should warn once total
                graph_mod._node2vec_embeddings({"a": {"b"}, "b": {"a"}}, dimensions=4)
                graph_mod._node2vec_embeddings({"a": {"b"}, "b": {"a"}}, dimensions=4)
                graph_mod._node2vec_embeddings({"a": {"b"}, "b": {"a"}}, dimensions=4)

            warns = [m for m in caplog.messages if "gensim" in m]
            assert len(warns) == 1, f"Expected 1 gensim warning, got {len(warns)}"


# ============================================================
# Fix #4: feature_report() includes knn_rate_* features
# ============================================================


class TestFeatureReportIncludesKnnRate:
    @pytest.mark.asyncio
    async def test_frozen_graph_report_lists_knn_rate(self) -> None:
        """After freeze(target_columns=...), report should list knn_rate_* features."""
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
            kg.freeze(target_columns=["products"], k=3)
            report = kg.feature_report()
            assert report["features"]["knn_rate"], (
                "feature_report should list knn_rate_* features after freeze"
            )
            # All entries should be properly prefixed
            assert all(
                k.startswith("knn_rate_") for k in report["features"]["knn_rate"]
            )

    @pytest.mark.asyncio
    async def test_after_to_ml_dataset_report_lists_knn_rate(self) -> None:
        """to_ml_dataset() sets _last_knn_target_columns; report picks them up."""
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
            kg.compute_features()  # populates self._features
            kg.to_ml_dataset("products", target_columns=["products"], k=3)
            report = kg.feature_report()
            assert report["features"]["knn_rate"]


# ============================================================
# Fix #5: Degenerate community detection warning
# ============================================================


class TestDegenerateCommunityWarning:
    @pytest.mark.asyncio
    async def test_warns_when_communities_equal_entities(self) -> None:
        """292 communities for 292 entities → degenerate, must warn."""
        # Records with no shared categorical/list values → no entity-entity
        # density via projection → Louvain produces ~1 community per entity.
        records = [{"id": f"E{i}", "category": f"unique_{i}"} for i in range(8)]

        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["category"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.compute_features()
            warnings_text = "\n".join(kg.build_warnings)
            # Either the "no communities" or "1 per entity" message is acceptable
            # — both indicate insufficient entity-entity topology.
            assert (
                "no communities" in warnings_text.lower()
                or "degenerate" in warnings_text.lower()
                or "1 per entity" in warnings_text
            )
            # Must always recommend the actual fix
            assert "add_relationship" in warnings_text or "relationship_columns" in warnings_text

    @pytest.mark.asyncio
    async def test_no_false_warning_on_healthy_communities(self) -> None:
        """If communities < entities by enough margin, no warning."""
        # Many shared values → real community structure
        records = [
            {"id": f"E{i}", "category": "A" if i < 5 else "B"} for i in range(10)
        ]

        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["category"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            kg.compute_features()
            assert not any(
                "degenerate" in w.lower() for w in kg.build_warnings
            )


# ============================================================
# Fix #6: to_cytoscape(focus=...) fallback when entity not found
# ============================================================


class TestFocusFallback:
    @pytest.mark.asyncio
    async def test_unresolved_focus_falls_back_to_default(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """to_cytoscape(focus='nonexistent') should warn + return default selection."""
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
            with caplog.at_level(
                logging.WARNING, logger="pydantic_ai_cloudflare.visualization"
            ):
                spec = kg.to_cytoscape(focus="DOES_NOT_EXIST", hops=2)
            # Was 0 in v0.2.1; now must return real graph
            assert spec["nodes"], "expected non-empty fallback subgraph"
            assert any(
                "did not resolve" in m for m in caplog.messages
            ), caplog.messages

    @pytest.mark.asyncio
    async def test_resolved_focus_still_works(self) -> None:
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
            spec = kg.to_cytoscape(focus="A0", hops=1)
            assert spec["nodes"]
            ids = {n["data"]["id"] for n in spec["nodes"]}
            assert any("a0" in i for i in ids)


# ============================================================
# Fix #7: raw_data truncation + opt-out
# ============================================================


class TestRawDataHandling:
    @pytest.mark.asyncio
    async def test_raw_data_truncated_by_default(self) -> None:
        """Long string values should be truncated to 200 chars by default."""
        big_text = "x" * 1000
        records = [
            {"id": f"E{i}", "industry": "SaaS", "notes": big_text} for i in range(3)
        ]
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            spec = kg.to_cytoscape()
            entity_nodes = [n for n in spec["nodes"] if n["data"]["type"] == "entity"]
            assert entity_nodes
            for n in entity_nodes:
                rd = n["data"].get("raw_data", {})
                if "notes" in rd:
                    assert len(rd["notes"]) <= 201  # 200 + ellipsis
                    assert rd["notes"].endswith("…")

    @pytest.mark.asyncio
    async def test_include_raw_data_false_drops_field(self) -> None:
        """include_raw_data=False removes raw_data entirely from output."""
        records = [{"id": f"E{i}", "industry": "SaaS"} for i in range(3)]
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            spec = kg.to_cytoscape(include_raw_data=False)
            for n in spec["nodes"]:
                assert "raw_data" not in n["data"]

    @pytest.mark.asyncio
    async def test_raw_data_max_chars_none_disables_truncation(self) -> None:
        """raw_data_max_chars=None preserves legacy v0.2.0 behavior."""
        big_text = "y" * 800
        records = [{"id": "E0", "industry": "SaaS", "notes": big_text}]
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                records,
                id_column="id",
                categorical_columns=["industry"],
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            spec = kg.to_cytoscape(raw_data_max_chars=None)
            entity = next(n for n in spec["nodes"] if n["data"]["type"] == "entity")
            assert entity["data"]["raw_data"]["notes"] == big_text


# ============================================================
# HTML viz UX overhaul (interactive filters, isolation, hover)
# ============================================================


class TestHtmlVizUX:
    """Smoke-tests the rendered HTML contains the new interactive controls.

    We can't run a real browser here, but we can assert the HTML payload
    contains the markers for each new feature so a regression that drops
    them from the template fails loudly.
    """

    @pytest.mark.asyncio
    async def test_html_contains_filter_sections(self) -> None:
        with patch.dict(os.environ, _env()):
            kg = EntityGraph()
            await kg.build_from_records(
                RECORDS,
                id_column="id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                relationship_columns={"competitor": "COMPETES_WITH"},
                text_columns=[],
                extract_entities=False,
                compute_similarity=False,
            )
            html = kg.render_html()
            # Filter sections by ID
            for marker in (
                'id="edge-filters"',
                'id="node-filters"',
                'id="community-filter-section"',
                'data-action="all"',
                'data-action="none"',
            ):
                assert marker in html, f"Missing filter marker: {marker}"

    @pytest.mark.asyncio
    async def test_html_contains_isolate_button(self) -> None:
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
            html = kg.render_html()
            for marker in (
                'id="action-isolate"',
                'id="selection-section"',
                'id="action-clear"',
            ):
                assert marker in html, f"Missing selection marker: {marker}"

    @pytest.mark.asyncio
    async def test_html_contains_display_toggles(self) -> None:
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
            html = kg.render_html()
            for marker in (
                'id="toggle-edge-labels"',
                'id="toggle-hover-highlight"',
                'id="toggle-bold-edges"',
                'id="toggle-arrows"',
            ):
                assert marker in html, f"Missing toggle marker: {marker}"

    @pytest.mark.asyncio
    async def test_html_contains_keyboard_shortcuts_help(self) -> None:
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
            html = kg.render_html()
            # The help line names every shortcut
            assert "<kbd>/</kbd>" in html
            assert "<kbd>Esc</kbd>" in html
            assert "<kbd>e</kbd>" in html
            assert "<kbd>f</kbd>" in html

    @pytest.mark.asyncio
    async def test_html_status_bar_present(self) -> None:
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
            html = kg.render_html()
            assert 'id="status-bar"' in html
            assert 'id="toolbar"' in html
            # Toolbar buttons
            assert 'data-toolbar="fit"' in html
            assert 'data-toolbar="reset"' in html

    @pytest.mark.asyncio
    async def test_html_writes_to_file(self, tmp_path) -> None:
        """Sanity: file written to disk contains all the new structure."""
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
            path = tmp_path / "graph.html"
            html = kg.render_html(str(path), title="MyGraph")
            assert path.exists()
            content = path.read_text()
            assert content == html
            assert "MyGraph" in content
            # Sample of new content:
            assert "Edge types" in content
            assert "Display" in content


# ============================================================
# Fix #8: generate_feature_from_text transparency + degenerate-warn
# ============================================================


class TestGenerateFeatureFromTextTransparency:
    """Tests verb mode + degenerate detection.

    We don't run the LLM here — we monkeypatch ``cf_structured`` to return
    a fixed FeatureSpec, then check the wrapper's warning + verbose output.
    """

    @pytest.mark.asyncio
    async def test_verbose_mode_returns_metadata(self, monkeypatch) -> None:
        from pydantic_ai_cloudflare import feature_engine, structured

        async def fake_cf_structured(prompt, model_cls, *, model=None, **kwargs):
            return model_cls(
                feature_name="my_feature",
                computation="adoption_rate",
                node_type="entity",
                target_value="zero trust",
                reference_filter_column="",
                reference_filter_value="",
            )

        def fake_compute_feature(graph, **kwargs):
            return {"E0": 0.5, "E1": 0.7, "E2": 0.0}

        # cf_structured is imported INSIDE generate_feature_from_text,
        # so patch the source module — the import resolves at call time.
        monkeypatch.setattr(structured, "cf_structured", fake_cf_structured)
        monkeypatch.setattr(feature_engine, "compute_feature", fake_compute_feature)

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
            result = await feature_engine.generate_feature_from_text(
                kg, "How many peers have zero trust?", verbose=True
            )
            assert isinstance(result, dict)
            assert "feature" in result
            assert "computation_type_used" in result
            assert result["computation_type_used"] == "adoption_rate"
            assert result["feature_name"] == "my_feature"
            assert result["target_value"] == "zero trust"
            assert "degenerate" in result

    @pytest.mark.asyncio
    async def test_degenerate_output_warns(
        self,
        monkeypatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from pydantic_ai_cloudflare import feature_engine, structured

        async def fake_cf_structured(prompt, model_cls, *, model=None, **kwargs):
            return model_cls(
                feature_name="all_zeros",
                computation="shared_count",
                node_type="entity",
                target_value="",
                reference_filter_column="",
                reference_filter_value="",
            )

        # Return all-zero values — should trigger the warning
        def fake_compute_feature(graph, **kwargs):
            return {f"E{i}": 0.0 for i in range(10)}

        monkeypatch.setattr(structured, "cf_structured", fake_cf_structured)
        monkeypatch.setattr(feature_engine, "compute_feature", fake_compute_feature)

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
            with caplog.at_level(
                logging.WARNING, logger="pydantic_ai_cloudflare.feature_engine"
            ):
                result = await feature_engine.generate_feature_from_text(
                    kg, "useless feature", verbose=True
                )
            # Warning should be issued
            assert any(
                "degenerate" in m.lower() for m in caplog.messages
            ), caplog.messages
            # Verbose result should flag it
            assert result["degenerate"] is True

    @pytest.mark.asyncio
    async def test_legacy_return_shape_preserved(self, monkeypatch) -> None:
        """Default (verbose=False) still returns dict[str, float]."""
        from pydantic_ai_cloudflare import feature_engine, structured

        async def fake_cf_structured(prompt, model_cls, *, model=None, **kwargs):
            return model_cls(
                feature_name="x",
                computation="adoption_rate",
                node_type="entity",
                target_value="x",
                reference_filter_column="",
                reference_filter_value="",
            )

        def fake_compute_feature(graph, **kwargs):
            return {"E0": 0.5, "E1": 0.7}

        monkeypatch.setattr(structured, "cf_structured", fake_cf_structured)
        monkeypatch.setattr(feature_engine, "compute_feature", fake_compute_feature)

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
            result = await feature_engine.generate_feature_from_text(
                kg, "test", verbose=False
            )
            # Plain dict[str, float], not the verbose envelope
            assert all(isinstance(v, float) for v in result.values())
            assert "feature" not in result  # Only present in verbose mode
