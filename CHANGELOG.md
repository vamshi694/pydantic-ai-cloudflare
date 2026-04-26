# Changelog

## 0.1.0 (2026-04-26)

Initial release.

- **CloudflareProvider** — Workers AI model provider via OpenAI-compatible API. AI Gateway auto-routing with custom metadata.
- **BrowserRunToolset** — 6 web interaction tools (browse, extract, crawl, scrape, discover_links, screenshot) via Browser Run REST API.
- **CloudflareEmbeddingModel** — Workers AI text embeddings (bge-base-en-v1.5).
- **VectorizeToolset** — RAG tools (search_knowledge, store_knowledge) backed by Vectorize + Workers AI embeddings.
- **D1MessageHistory** — Conversation persistence using D1 serverless SQLite. Auto-creates tables.
- **GatewayObservability** — AI Gateway log queries, analytics, and feedback API.
