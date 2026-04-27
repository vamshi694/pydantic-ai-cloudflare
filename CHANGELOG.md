# Changelog

All notable changes to this project will be documented in this file.

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
