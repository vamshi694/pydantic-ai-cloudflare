# Changelog

All notable changes to this project will be documented in this file.

## 0.2.5 (2026-04-29)

### Docs
- **README: rewritten structured output section** — explains why tool calling fails on Workers AI, how `cf_structured()` solves it, comparison table, all options (AI Gateway, caching, prompt caching, retries).
- **Notebook 08: Structured Output Deep Dive** — simple → complex schema examples with real outputs, AI Gateway caching demo, prompt caching, schema complexity check. Shows the exact pattern that fails with LangChain `with_structured_output()` working with `cf_structured()`.

## 0.2.4 (2026-04-28)

### Added
- **AI Gateway support on `cf_structured()`** — route structured output calls through AI Gateway for logging, analytics, and caching. Same as langchain-cloudflare's AI Gateway integration.
- **Response caching** — `cache_ttl=300` caches identical prompts for 5 minutes via AI Gateway. `skip_cache=True` to bypass. `cache_key="my-key"` for custom cache keys.
- **Prompt caching** — `session_id="sess-123"` routes to same model instance for KV prefix cache hits. Reduces latency on multi-turn conversations.
- **Gateway request handling** — `gateway_timeout`, `gateway_max_attempts`, `gateway_retry_delay`, `gateway_backoff` for gateway-level retries and timeouts.
- **Gateway metadata** — `gateway_metadata={"user_id": "u-123"}` attaches custom metadata to every gateway log entry.

## 0.2.3 (2026-04-27)

A correctness patch for two issues that survived v0.2.2.

No breaking changes.

### Fixed

- **`score_one()` feature parity (round 2).** v0.2.2 over-corrected the
  v0.2.1 undershoot: where v0.2.1 emitted *fewer* features than
  `to_ml_dataset()`, v0.2.2 ended up emitting *more* (`knn_max_distance`
  and the v0.2.0-era `knn_peers` snuck back in as scalar features the
  trained model never saw). Models trained on `to_ml_dataset()`'s
  union schema rejected v0.2.2's output for the same root cause.
  v0.2.3:
  - `freeze()` now snapshots the **full union of `to_ml_dataset()`'s
    keys** (not just `to_feature_dicts()`'s) — including `knn_rate_*`
    floats, `knn_avg_distance`, `knn_min_distance`.
  - `score_one()` filters its output to that schema, fills missing
    keys with 0, drops extras (specifically `knn_max_distance`).
  - `knn_peers` is preserved as the single non-feature key — a list of
    peer entity labels for explainability. Drop it before passing the
    dict to a model: `{k: v for k, v in scored.items() if k != "knn_peers"}`
    or, more robustly: `{k: v for k, v in scored.items() if isinstance(v, (int, float))}`.
  - Updated `score_one()` docstring to make the contract explicit.
  - End-to-end pipeline test (`test_inference_dict_feedable_to_model`)
    proves train→infer feature sets match exactly.
- **Friendlier warning when viz filters wipe out the whole subgraph.**
  Calling `to_cytoscape(exclude_node_types=[...])` with a list covering
  every type in the graph used to silently return `{nodes:[], edges:[]}`.
  Now logs a clear WARNING explaining what the filters did and which
  arguments to remove. (The earlier-reported `to_cytoscape(focus=...)`
  empty-result behavior was already fixed in v0.2.2; this hardens the
  remaining edge cases.)

### Tests

- +7 new tests in `tests/test_graph_v023.py` covering the round-2
  parity contract, `knn_max_distance` non-emission, `knn_peers`
  explainability preservation, end-to-end train→infer DataFrame
  compatibility, `score_batch` ↔ `score_one` consistency, and the
  filter-wipeout warning. 239 total passing.

## 0.2.2 (2026-04-27)

A bugfix release driven by real CF1 production data. Fixes 8 issues
surfaced during field testing — most critically `score_one()` was
emitting fewer features than `to_ml_dataset()`, silently breaking the
freeze/score pipeline that v0.2.0 was built around.

No breaking changes. All v0.2.1 code continues to work.

### Fixed (production blockers)

- **`score_one()` feature parity with `to_ml_dataset()`** — v0.2.1
  emitted 33 features per record while `to_ml_dataset()` produced 43,
  silently breaking ML inference (the trained model expected 43
  columns). `freeze()` now snapshots the full feature schema produced
  by `to_feature_dicts()`, and `score_one()` fills in any missing keys
  with 0. Adds the missing `IN_<COL>_RANGE_degree` (numeric) and
  `<REL_TYPE>_degree` (relationship) features that v0.2.1 only emitted
  for categorical and list columns.
- **`build_temporal_dataset()` no longer requires Cloudflare
  credentials** when LLM features are disabled. `EntityGraph.__init__`
  now defers credential resolution until the first real API call (via
  `_ensure_creds()`), so pure-structural workflows over CSVs work
  out-of-the-box without `CLOUDFLARE_ACCOUNT_ID`/`CLOUDFLARE_API_TOKEN`
  env vars. Real API calls (LLM extraction, embeddings, ask, etc.)
  still raise a clear `CloudflareConfigError` with the same message as
  before.
- **gensim "not installed" warning fires once per process** instead of
  3+ times per build. The module-level `_GENSIM_WARNED` flag was dead
  code — now correctly gates the warning at the only emission site.
- **`feature_report()` now lists `knn_rate_*` features.** v0.2.1 left
  this section empty because `knn_rate_*` features live in the
  return value of `knn_rate_features()` / `to_ml_dataset()` and never
  populate `self._features`. Now expanded from `_frozen_target_values`
  (frozen graph) or `_last_knn_target_columns` (after `to_ml_dataset()`).
- **Degenerate community detection now warns explicitly.** When Louvain
  produces ~1 community per entity (or 0 communities at all), the build
  emits a clear warning explaining the cause (insufficient
  entity-to-entity topology) and pointing to `add_relationship()` /
  `relationship_columns` as the fix. Surfaced via `build_warnings`,
  the logger, and `feature_report()`.

### Changed (HTML viz UX overhaul)

`render_html()` was hard to read on graphs with many edges. Total
rebuild of the in-page UX so users can actually understand large
graphs without leaning on click-to-inspect.

- **Edge-type filter checkboxes.** New "Edge types" section in the
  side panel with a checkbox per edge type (HAS_PRODUCT,
  COMPETES_WITH, USES_TECH, IN_<COL>_RANGE, etc.) showing color
  swatch and edge count. Click to show/hide an entire edge type.
  "All" / "None" buttons for bulk toggle.
- **Node-type filter checkboxes.** Same treatment for entity / concept
  / list / range / categorical node types — toggle whole categories
  on/off.
- **Community filter checkboxes** (when `color_by="community"`).
- **Click-to-isolate.** Click any node, an "Isolate" button appears in
  the side panel — hides everything except the selected node and its
  closed neighborhood. "Clear" / `Esc` / Reset toolbar button restores
  the full view.
- **Hover highlighting** (default ON, toggleable). Hovering a node
  fades unrelated nodes/edges to 8% opacity and highlights neighbors
  + connecting edges with the gold accent. Edge labels appear on the
  highlighted edges.
- **Edge label toggle** (`e` key). Show/hide labels on every edge.
- **Bolder edges by default** — width 2.2-5px (was 1-4px), opacity
  0.78 (was 0.6) so the structure is readable at a glance. Toggle to
  thin mode if you prefer.
- **Arrow toggle** for directional edge types.
- **Live "X / Y visible"** counts in the Stats section update as
  filters change.
- **Search-match glow.** Matching nodes get a gold border, neighbors
  stay bright, the rest fades — works alongside filters and isolation.
- **Selection panel.** Info panel shows connections grouped by edge
  type ("HAS_PRODUCT (3) · COMPETES_WITH (1) · USES_TECH (5)") and
  has a close button.
- **Status bar** shows transient feedback ("Isolated AcmeCorp", "3
  matches for 'zscaler'") with `Esc` to clear.
- **Toolbar** with Fit / Reset buttons in the corner.
- **Keyboard shortcuts**: `/` focuses search, `Esc` resets all, `e`
  toggles edge labels, `f` fits view. Help line shown in the side
  panel header.
- **Wider sidebar** (320px from 280px) and collapsible
  `<details>`-based sections so the panel stays organized as it grows.

No API change for `kg.render_html(...)` — every existing arg still
works. The improvements are entirely in the rendered HTML/JS.

### Fixed (UX)

- **`to_cytoscape(focus="UnknownEntity", ...)` no longer returns an
  empty graph.** The unresolved-focus path silently produced
  `{nodes: [], edges: []}` in v0.2.1. Now logs a warning and falls
  back to the default top-N-by-degree selection so users see *something*
  and can fix their focus argument. Isolated focus nodes (no neighbors
  within the hop budget) also warn.
- **HTML/D3/Cytoscape output truncates long string fields by default.**
  v0.2.1 dumped the full source record into every entity node's
  `raw_data`, producing 100KB+ payloads when records contained long
  `ae_notes` / description fields. New `raw_data_max_chars=200`
  default truncates with an ellipsis. New `include_raw_data=False`
  drops the field entirely. Set `raw_data_max_chars=None` for the
  legacy v0.2.0 behavior.
- **`generate_feature_from_text()` now reports its decision.** New
  `verbose=True` returns the chosen computation type, target value,
  and reference filter alongside the feature dict so you can audit
  what the LLM did. Always logs a WARNING when ≥80% of values are
  identical (a sure sign the LLM picked the wrong computation and the
  feature is degenerate). Default return shape unchanged
  (`dict[entity_label, float]`).

## 0.2.1 (2026-04-27)

A UX-focused patch release. No breaking changes — all v0.2.0 code continues
to work. Driven by a first-time-user audit that flagged onboarding friction:
"powerful but steep learning curve."

### Added

- **`EntityGraph.quick_build()`** — one-line graph builder for the 80% use
  case. Auto-profiles the dataset and turns OFF expensive options
  (LLM extraction, all-pairs similarity) by default. Goes from CSV to
  ML features in 4 lines:
  ```python
  kg = EntityGraph()
  await kg.quick_build(records, id_column="customer_id")
  features = kg.to_feature_dicts()
  ```
- **`GraphConfig` dataclass** — bundle the 20+ parameters of
  `build_from_records()` into one named object. Exported from the package
  root. Use with `kg.build_from_config(records, config)`.
- **`EntityGraph.build_warnings`** — read-only property exposing build-time
  warnings (high null rates, sentinel zeros, low-cardinality columns)
  collected during the most recent `build_from_records()` call.
- **Usable repr / `len` / `in` / iteration** on `EntityGraph`:
  ```python
  >>> kg
  <EntityGraph name='default' entities=10000 nodes=28705 edges=153642>
  >>> len(kg)
  10000
  >>> "AcmeCorp" in kg
  True
  >>> for label in kg: ...
  ```

### Changed

- **Build warnings are now logged automatically** at `WARNING` level via
  the `pydantic_ai_cloudflare.graph` logger (previously stored silently
  on `_build_warnings` and only surfaced via `feature_report()`). Users
  who run `logging.basicConfig(level=logging.WARNING)` will see all
  data-quality warnings as the build runs.
- **Better error messages** for the freeze/score lifecycle. The
  `RuntimeError` raised from `score_one()` on an unfrozen graph now
  explains the *why* (training/inference parity, no feature drift) and
  points to alternatives (`compute_features()`, `to_feature_dicts()`)
  for users who don't actually need freeze. Same treatment for
  `add_records()`, `build_from_records()`, and the metadata-missing
  error from `_record_to_feature_nodes()`.
- **README** restructured for first-time users: new `quick_build()`
  example at the top of the EntityGraph section, new "Troubleshooting"
  section covering the 9 most common errors and how to fix them, new
  `GraphConfig` example.

## 0.2.0 (2026-04-27)

A correctness + visualization release driven by real testing on customer data.
Focused on the EntityGraph: making it trustworthy for production ML and useful
for LLM-driven exploration. No GNN, no triple stores — just the same utility
library, made robust.

### Fixed (production correctness from CF1 testing)
- **`find_similar()` no longer dominated by hub features.** New `use_idf=True` (default) discounts paths through hub feature nodes using `1/log(degree+1)`. Sentinel-zero columns (e.g. `propensity_score=0` connecting 170 accounts) no longer drive fake similarity. Works automatically — no more manual `edge_type_weights` for the common case.
- **Multi-path similarity.** `find_similar()` now accumulates score across every distinct path source→feature→target, instead of stopping after the first path. The `via` list now shows ALL shared features (industry, AWS, K8s, CDN, WAF), not just the first.
- **Sentinel-zero numeric columns auto-detected.** When a numeric column has ≥30% zero values (configurable via `sentinel_zero_rate`), zeros are treated as missing and excluded from the graph. User can override with `sentinel_zero_columns=[]`. Catches the `propensity_score=0` footgun before it pollutes features.
- **`co_occurrence_features()` `min_support` parameter** (default 5). Pairs with co_count below this threshold are dropped. Prevents statistically fragile lift scores driven by 1-2 accounts. Pass `min_support=1` for legacy behavior.
- **`explain()` shared_nodes labels resolve correctly.** Now shows `propensity_score=zero` instead of bare `'zero'`. Range nodes filtered by default (`include_range_nodes=False`); set True to keep them. Output is finally human-presentable.
- **`save_features()` warns when target_columns are missing** but `knn_rate_*` features were computed earlier in the session. Catches the silent "saved 16 instead of 49 features" bug. Saved JSON now includes `snapshot_date`, `target_columns`, `k`, and `build_meta` for reproducibility.

### Added (point-in-time, freeze, viz)
- **Point-in-time features (`time_column` + `as_of`).** No more training-time data leakage. `kg.build_from_records(records, time_column='created_at', as_of='2024-01-31')` only includes records on/before the cutoff. Snapshot date is persisted in `kg._snapshot_date` and saved with features.
- **`build_temporal_dataset()`** module-level helper. Build a leakage-free training set across multiple snapshot dates with forward-window labels:
  ```python
  X, y = await build_temporal_dataset(
      records, id_column='account_id', time_column='event_date',
      snapshot_dates=['2023-01-01', '2023-04-01', '2023-07-01'],
      label_column='purchased_casb_at', label_horizon_days=180,
      target_columns=['products_owned'],
  )
  ```
- **`freeze()` + `score_one()` + `score_batch()`.** Lock the graph topology after training and score new records without mutating any state — `add_records()` raises after freeze. Eliminates "feature drift" between training and inference. The frozen graph caches communities, peer-assignment rules, and target-value catalog so scoring uses exactly the same topology that produced the training features.
- **Parallel LLM extraction** in `build_from_records()`. New `llm_concurrency` parameter (default 8) batches all `extract_entities` / `extract_relationships` / `summarize_text` jobs through `asyncio.gather` + Semaphore. 300 records went from ~30 minutes sequential to ~3 minutes concurrent.
- **Build-time profiling.** New `profile=True` (default) flags low-cardinality, high-null, and other footgun columns at build time. Surfaced via `kg._build_warnings` and `feature_report()`.
- **`feature_report()` and `print_report()`.** Self-describing summary of what the graph contains, which features were generated, what warnings were raised, and whether PageRank is meaningful here. Helps users debug their build instead of staring at silent feature drift.
- **`compute_features(community_method='entity_projection')`.** New default community method projects the bipartite graph onto entity-only space (Adamic-Adar weighted) before running Louvain. Produces ~25 communities on 300 entities instead of ~5 from naive bipartite Louvain. Pass `'bipartite'` for the legacy behavior.

### Added (visualization — `GraphVisualizer`)
- **Interactive HTML rendering.** `kg.render_html('graph.html', color_by='community')` produces a self-contained HTML file (Cytoscape.js via CDN, zero local deps) with pan/zoom/drag, search, layout switching, info panel on click, and a community-color legend.
- **Cytoscape.js JSON export.** `kg.to_cytoscape()` returns embeddable JSON for any web frontend.
- **D3.js force-graph JSON.** `kg.to_d3_json()`.
- **Mermaid flowchart.** `kg.to_mermaid()` for documentation, GitLab/Confluence/notebooks.
- **GraphML XML.** `kg.to_graphml('graph.graphml')` for Gephi or yEd.
- **Color coding.** `color_by='community'` (default if computed), `'type'`, or any entity-data column name. Edge types are color-coded too: SIMILAR_TO dashed, entity-to-entity (COMPETES_WITH etc.) red, feature edges muted gray. Edge width ∝ weight.
- **Subgraph selection.** `max_nodes`, `focus`, `hops`, `include_node_types`, `exclude_node_types` for rendering manageable subgraphs of large graphs.

### Changed
- `EntityGraph.__init__` now stores `_build_meta` capturing the column logic from `build_from_records()`. This is what makes `score_one()` reproducible.
- `EntityGraph` now exposes `is_frozen`, `unfreeze()`.
- `min_support=5` is the new default for `co_occurrence_features` (was unbounded). Pass `min_support=1` if you really want every pair.
- Public API exports: `EntityGraph`, `KnowledgeGraph` (alias), `build_temporal_dataset`, `GraphVisualizer`.

## 0.1.9 (2026-04-27)

### Fixed (from production usage feedback)
- **`find_similar()` now supports `edge_type_weights` and `exclude_edge_types`** — numeric bucket nodes (propensity_score_range:tiny) no longer drown out product/competitor signal. Weight product edges 3-5x higher.
- **`explain()` shared nodes show typed labels** — `tech:aws` instead of raw node IDs. Human-presentable output.
- **`to_ml_dataset()` y labels canonicalized** — `magic wan / magic firewall` and `magic wan` collapse to one label column, not two.
- **`recommend()` `exclude_dominant` parameter** — skip values present in >70% of entities (too common to recommend). Prevents Zero Trust (79% prevalence) from blocking all recommendations.
- **Louvain resolution=1.5** — finds more granular communities (was finding only 5 for 292 accounts across 74 industries).

## 0.1.8 (2026-04-27)

### Fixed (all 11 bugs from team review)
- **`save_features()`** now saves the full feature pipeline (knn_rate_*, knn distances) when `target_columns` is passed. Models trained on `to_ml_dataset()` can now be reproduced at inference.
- **`profile_data`** text threshold lowered 80→50 so short descriptions aren't misclassified.
- **`profile_data`** auto-skips URLs, phone numbers, and UUIDs (regex patterns) instead of making them graph nodes.
- **`find_similar()`** filters results to only entities from original `build_from_records()` rows — relationship targets (competitors) no longer appear.
- **`recommend()`** returns a dict with `entity_found`, `peers_found`, `own_values`, and `reason` instead of a silent empty list.
- **`co_occurrence_features()`** warns when `outcome_filter` is set but `outcome_column` wasn't configured.
- **`compute_feature()`** is now a method on `EntityGraph` — `kg.compute_feature(...)` works directly.
- **gensim warning** fires once only, not on every `compute_features()` call.
- **`explain()`** accepts `min_rate` parameter (was hardcoded 0.2) and includes `recommendation_reason` field.
- **Phone regex** tightened to require `+/(` prefix — dates like `2018-03-15` no longer match.

## 0.1.7 (2026-04-27)

### Added
- **`explain(entity_id)`** — traces actual graph paths: community membership, peer shared nodes, per-recommendation evidence, entity-to-entity relationships.
- **`dd.review()`** — shows profile guesses with ⚠️ flags for uncertain classifications.
- **Progress logging** — `build_from_records` and `compute_features` log at 10% intervals for large datasets.

## 0.1.6 (2026-04-27)

### Changed
- **Renamed `KnowledgeGraph` → `EntityGraph`** — honest about what it is. It's a bipartite feature graph with entity-to-entity relationship support, not a full knowledge graph with ontologies and inference. `KnowledgeGraph` kept as backward-compatible alias.

### Added
- **`add_relationship()`** — direct entity-to-entity typed edges. `kg.add_relationship("Cisco", "COMPETES_WITH", "Zscaler")`. Enables multi-hop traversal that bipartite graphs can't express.
- **`relationship_columns`** on `build_from_records()` — automatically creates entity-to-entity edges from structured columns like `primary_competitor`, `partner`, `referred_by`.
- **`extract_relationships=True`** — LLM extracts typed relationships (COMPETES_WITH, DISPLACED, ACQUIRED, etc.) from text columns via Workers AI.
- **`min_group_size`** on `auto_canonicalize()` — controls merging aggressiveness. Prevents over-merging distinct entities like "Magic WAN" and "Magic Transit".

### Fixed
- **Node2Vec hash fallback removed** — was generating identical values for all entities (silent noise). Now returns empty dict if gensim not installed, with a clear warning.
- **PageRank warning** — docstring now states it measures feature-mediated centrality on bipartite graphs, not entity importance. Recommends `community_id` and `knn_rate_*` for meaningful ML features.
- **`auto_canonicalize` CAUTION** — docstring warns about over-merging risk.

## 0.1.5 (2026-04-27)

### Fixed
- Published `auto_canonicalize` and OSS hygiene files that were missing from v0.1.4 PyPI package.

## 0.1.4 (2026-04-27)

### Added
- **Canonicalization** — `canonical_map` parameter on `KnowledgeGraph()` collapses entity aliases (Zscaler/ZS/zscaler inc → one node). Manual dict or LLM-powered via `kg.auto_canonicalize()`.
- **Outcome awareness** — `outcome_column` on `build_from_records()`. Every edge carries an outcome. `co_occurrence_features()` returns `co_won` and `co_rate_won` alongside P(B|A).
- **Lift** — `co_occurrence_features()` now returns `lift = P(A∧B)/(P(A)·P(B))`. Lift > 1.5 = real signal.
- **Edge confidence** — `confidence_map` per column type. Structured fields get 1.0, LLM-extracted get 0.7. Multi-source edges accumulate.
- **`to_ml_dataset()`** — Returns `(X, y)` with all graph features + binary labels per value. Ready for sklearn/XGBoost.
- **`save_features()` / `load_features()`** — Persist feature matrix to JSON for reproducible inference.
- **`compute_feature()`** — Custom per-entity features from graph with 7 computation types (shared_count, overlap_rate, adoption_rate, distance_to_nearest, subgraph_degree, reference_rate, co_occurrence_lift).

## 0.1.3 (2026-04-27)

### Added
- **Louvain community detection** — Replaces label propagation. Finds real clusters (29 communities from 10K accounts vs 1 before).
- **Node2Vec embeddings** — Biased random walks + gensim Word2Vec. Scales to 28K+ nodes. Adds `n2v_norm`, `n2v_avg_neighbor_dist` features.
- **Temporal decay** — `temporal_column` + `temporal_decay` on `build_from_records()`. Edge weight = exp(-λ × days_ago). Recent: 0.95, two-year-old: 0.05.

### Benchmarks
- 10,000 rows: 28,705 nodes, 153,642 edges in 0.25s. 29 Louvain communities. 18x temporal recency boost.

## 0.1.2 (2026-04-27)

### Added
- **`KnowledgeBase`** — Managed RAG via AI Search with hybrid search (semantic + BM25), reranking, relevance boosting.
- **`DIYKnowledgeBase`** — Vectorize + Workers AI. `ingest()` accepts URLs, local files, directories, glob patterns. Built-in chunker, batch embedding, reranking with bge-reranker-base.
- **`KnowledgeGraph`** — Typed knowledge graph from tabular data with ML feature engineering. Graph construction, PageRank, clustering, KNN rates, recommendations, co-occurrence.
- **`profile_data()`** — Auto-classifies columns (id, text, categorical, numeric, list, date, skip).

## 0.1.1 (2026-04-26)

### Added
- **`KnowledgeBase`** — First version of managed RAG.
- **`DIYKnowledgeBase`** — First version of DIY RAG with Vectorize.

## 0.1.0 (2026-04-26)

Initial release.

- **`CloudflareProvider`** — Workers AI model provider via OpenAI-compatible API. AI Gateway auto-routing.
- **`cloudflare_model()`** / **`cloudflare_agent()`** — Convenience functions for creating agents.
- **`cf_structured()`** — Complex structured output on all 6 Workers AI models (Llama, Qwen, Kimi, Gemma, GLM, DeepSeek).
- **`BrowserRunToolset`** — 6 web tools (browse, extract, crawl, scrape, discover_links, screenshot).
- **`CloudflareEmbeddingModel`** — Workers AI text embeddings.
- **`VectorizeToolset`** — RAG search + store tools.
- **`D1MessageHistory`** — Conversation persistence via D1.
- **`GatewayObservability`** — AI Gateway logs, analytics, feedback.
- **`list_models()`** / **`recommend_model()`** — Model catalog and recommendations.
- **`schema_stats()`** / **`simplify_schema()`** — Schema optimization for structured output.
