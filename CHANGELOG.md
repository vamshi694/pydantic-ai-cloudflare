# Changelog

All notable changes to this project will be documented in this file.

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
