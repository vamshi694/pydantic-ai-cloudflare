"""KnowledgeBase — the 4-line RAG experience.

Two paths to RAG on Cloudflare:

1. KnowledgeBase (recommended): Uses AI Search with hybrid search
   (semantic + BM25), reranking, query rewriting, and relevance boosting.
   Cloudflare manages the full pipeline.

2. DIYKnowledgeBase: Uses Vectorize + Workers AI embeddings directly.
   You control chunking, embedding model, and metadata. Add reranking
   on top via Workers AI reranker.

Usage (path 1 -- managed):

    from pydantic_ai_cloudflare import KnowledgeBase

    kb = KnowledgeBase("my-docs")
    results = await kb.search("How does caching work?")
    answer = await kb.ask("How does caching work?")

Usage (path 2 -- DIY):

    from pydantic_ai_cloudflare import DIYKnowledgeBase

    kb = DIYKnowledgeBase(index_name="my-vectors")
    await kb.ingest(["https://docs.example.com"])
    results = await kb.search("How does caching work?", rerank=True)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response

logger = logging.getLogger(__name__)


# -- Simple text chunker (no external deps) --


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries.

    Tries to split on paragraph breaks, then sentences, then words.
    No external dependencies -- pure Python.

    Args:
        text: The text to chunk.
        chunk_size: Target chunk size in characters.
        overlap: Overlap between consecutive chunks.
        separators: Custom separators to try (in priority order).

    Returns:
        List of text chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    if separators is None:
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Try to break at a natural boundary
            best_break = -1
            for sep in separators:
                # Look backwards from the end for a separator
                pos = text.rfind(sep, start + overlap, end)
                if pos > start:
                    best_break = pos + len(sep)
                    break

            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, accounting for overlap
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks


# ==========================================================
# Path 1: KnowledgeBase (managed via AI Search)
# ==========================================================


class KnowledgeBase:
    """Managed RAG using Cloudflare AI Search.

    AI Search handles chunking, embedding, indexing, hybrid search
    (semantic + BM25), reranking, query rewriting, and response
    generation. You just point it at your data and query.

    Prereqs:
        1. Create an AI Search instance in the dashboard
           (AI → AI Search → Create)
        2. API token with AI Search Edit + Run permissions

    Args:
        instance_name: AI Search instance name.
        account_id: Cloudflare account ID.
        api_key: API token.
        retrieval_type: "hybrid" (default), "vector", or "keyword".
        reranking: Enable cross-encoder reranking.
        reranking_model: Reranker model ID.
        max_results: Max search results.
        match_threshold: Minimum relevance score (0-1).
        context_expansion: Include N surrounding chunks (0-3).
        request_timeout: HTTP timeout.
    """

    def __init__(
        self,
        instance_name: str,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        retrieval_type: str = "hybrid",
        reranking: bool = True,
        reranking_model: str = "@cf/baai/bge-reranker-base",
        max_results: int = 10,
        match_threshold: float = 0.4,
        context_expansion: int = 1,
        boost_by: list[dict[str, str]] | None = None,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._instance = instance_name
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._retrieval_type = retrieval_type
        self._reranking = reranking
        self._reranking_model = reranking_model
        self._max_results = max_results
        self._match_threshold = match_threshold
        self._context_expansion = context_expansion
        self._boost_by = boost_by
        self._timeout = request_timeout
        self._headers = build_headers(self._api_key)

    def _url(self, endpoint: str) -> str:
        return (
            f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}"
            f"/ai-search/instances/{self._instance}/{endpoint}"
        )

    def _search_options(self) -> dict[str, Any]:
        """Build the ai_search_options dict for requests."""
        opts: dict[str, Any] = {
            "retrieval": {
                "retrieval_type": self._retrieval_type,
                "max_num_results": self._max_results,
                "match_threshold": self._match_threshold,
                "context_expansion": self._context_expansion,
            },
        }
        if self._reranking:
            opts["reranking"] = {
                "enabled": True,
                "model": self._reranking_model,
            }
        if self._boost_by:
            opts["retrieval"]["boost_by"] = self._boost_by
        return opts

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the knowledge base.

        Returns ranked chunks with source attribution, scores, and
        scoring details (vector_score, keyword_score, reranking_score).

        Args:
            query: Search query.
            max_results: Override default max results.
            filters: Metadata filters.

        Returns:
            List of result dicts with text, source, score, scoring_details.
        """
        body: dict[str, Any] = {
            "query": query,
            "ai_search_options": self._search_options(),
        }
        if max_results:
            body["ai_search_options"]["retrieval"]["max_num_results"] = max_results
        if filters:
            body["ai_search_options"]["retrieval"]["filters"] = filters

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("search"),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        result = data.get("result", {})
        return result.get("data", result.get("chunks", []))

    async def ask(
        self,
        question: str,
        *,
        system_prompt: str | None = None,
        max_results: int | None = None,
    ) -> str:
        """Ask a question and get an AI-generated answer with citations.

        Uses AI Search's /chat/completions endpoint which searches,
        retrieves relevant chunks, and generates an answer.

        Args:
            question: The question.
            system_prompt: Override default system prompt.
            max_results: Override max results for retrieval.

        Returns:
            AI-generated answer string.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        body: dict[str, Any] = {
            "messages": messages,
            "ai_search_options": self._search_options(),
        }
        if max_results:
            body["ai_search_options"]["retrieval"]["max_num_results"] = max_results

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("chat/completions"),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)

        choices = data.get("choices", data.get("result", {}).get("choices", []))
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return str(data.get("result", ""))

    def search_sync(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Synchronous version of search()."""
        import asyncio

        return asyncio.run(self.search(query, **kwargs))

    def ask_sync(self, question: str, **kwargs: Any) -> str:
        """Synchronous version of ask()."""
        import asyncio

        return asyncio.run(self.ask(question, **kwargs))


# ==========================================================
# Path 2: DIYKnowledgeBase (Vectorize + embeddings + reranker)
# ==========================================================


class DIYKnowledgeBase:
    """DIY RAG using Vectorize + Workers AI embeddings + reranker.

    Full control over chunking, embedding model, metadata, and
    retrieval strategy. Handles the embed → store → search → rerank
    pipeline.

    Prereqs:
        1. Create a Vectorize index:
           npx wrangler vectorize create NAME --dimensions 768 --metric cosine
        2. API token with Workers AI Read + Vectorize Edit

    Args:
        index_name: Vectorize index name.
        embedding_model: Workers AI embedding model.
        reranker_model: Workers AI reranker model (None to disable).
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        top_k: Default search results before reranking.
        rerank_top_k: Results to keep after reranking.
    """

    def __init__(
        self,
        index_name: str,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        embedding_model: str = "@cf/baai/bge-base-en-v1.5",
        reranker_model: str | None = "@cf/baai/bge-reranker-base",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        top_k: int = 20,
        rerank_top_k: int = 5,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._index = index_name
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._embedding_model = embedding_model
        self._reranker_model = reranker_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        self._timeout = request_timeout
        self._headers = build_headers(self._api_key)

    def _ai_url(self, model: str) -> str:
        return f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/run/{model}"

    def _vectorize_url(self, path: str) -> str:
        return (
            f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}"
            f"/vectorize/v2/indexes/{self._index}/{path}"
        )

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Workers AI."""
        # Batch into groups of 100 (API limit)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._ai_url(self._embedding_model),
                    headers=self._headers,
                    json={"text": batch},
                )
                resp.raise_for_status()
            data = resp.json()
            check_api_response(data)
            all_embeddings.extend(data.get("result", {}).get("data", []))
        return all_embeddings

    async def _rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Rerank documents using Workers AI reranker.

        Returns list of (original_index, score) sorted by relevance.
        """
        if not self._reranker_model or not documents:
            return [(i, 1.0) for i in range(len(documents))]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._ai_url(self._reranker_model),
                headers=self._headers,
                json={"query": query, "documents": documents},
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        results = data.get("result", [])

        # Sort by score descending
        ranked = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        return [(r.get("index", i), r.get("score", 0.0)) for i, r in enumerate(ranked)]

    async def ingest(
        self,
        sources: list[str],
        *,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Ingest content into the knowledge base.

        Accepts:
        - URLs (fetched via Browser Run, handles JS-rendered pages)
        - Local file paths (reads .txt, .md, .csv, .json, .py, etc.)
        - Directory paths (reads all supported files recursively)
        - Raw text strings

        Content is chunked, embedded, and stored in Vectorize.

        Args:
            sources: List of URLs, file paths, directory paths, or text.
            metadata: Extra metadata to attach to all chunks.

        Returns:
            Dict with stats: chunks_created, vectors_stored, sources_processed.
        """
        import glob
        import os

        from .browser_run import BrowserRunToolset

        # File extensions we can read as text
        _TEXT_EXTENSIONS = {
            ".txt",
            ".md",
            ".markdown",
            ".rst",
            ".csv",
            ".tsv",
            ".json",
            ".jsonl",
            ".yaml",
            ".yml",
            ".toml",
            ".py",
            ".js",
            ".ts",
            ".go",
            ".rs",
            ".java",
            ".rb",
            ".php",
            ".html",
            ".htm",
            ".xml",
            ".css",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".env",
            ".cfg",
            ".ini",
            ".conf",
            ".log",
            ".tex",
        }

        all_chunks: list[dict[str, Any]] = []
        sources_processed = 0

        for source in sources:
            # Detect source type
            if source.startswith("http://") or source.startswith("https://"):
                # URL → fetch via Browser Run
                ts = BrowserRunToolset(
                    account_id=self._account_id,
                    api_key=self._api_key,
                    tools=["browse"],
                )
                content = await ts._browse(source)
                source_label = source
                sources_processed += 1

            elif os.path.isdir(source):
                # Directory → read all text files recursively
                for root, _dirs, files in os.walk(source):
                    for fname in sorted(files):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in _TEXT_EXTENSIONS:
                            continue
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, encoding="utf-8", errors="ignore") as f:
                                file_content = f.read()
                        except (OSError, PermissionError):
                            logger.warning(f"Skipping unreadable file: {fpath}")
                            continue
                        if not file_content.strip():
                            continue
                        file_chunks = chunk_text(
                            file_content, self._chunk_size, self._chunk_overlap
                        )
                        for i, chunk in enumerate(file_chunks):
                            chunk_meta = {
                                "text": chunk[:1000],
                                "source": fpath,
                                "filename": fname,
                                "chunk_index": str(i),
                            }
                            if metadata:
                                chunk_meta.update(metadata)
                            all_chunks.append({"text": chunk, "metadata": chunk_meta})
                        sources_processed += 1
                continue  # already chunked, skip the common chunking below

            elif os.path.isfile(source):
                # Single file → read it
                try:
                    with open(source, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (OSError, PermissionError):
                    logger.warning(f"Skipping unreadable file: {source}")
                    continue
                source_label = source
                sources_processed += 1

            elif "*" in source or "?" in source:
                # Glob pattern → expand and read matching files
                matched = glob.glob(source, recursive=True)
                for fpath in sorted(matched):
                    if not os.path.isfile(fpath):
                        continue
                    try:
                        with open(fpath, encoding="utf-8", errors="ignore") as f:
                            file_content = f.read()
                    except (OSError, PermissionError):
                        continue
                    if not file_content.strip():
                        continue
                    file_chunks = chunk_text(file_content, self._chunk_size, self._chunk_overlap)
                    for i, chunk in enumerate(file_chunks):
                        chunk_meta = {
                            "text": chunk[:1000],
                            "source": fpath,
                            "filename": os.path.basename(fpath),
                            "chunk_index": str(i),
                        }
                        if metadata:
                            chunk_meta.update(metadata)
                        all_chunks.append({"text": chunk, "metadata": chunk_meta})
                    sources_processed += 1
                continue

            else:
                # Raw text string
                content = source
                source_label = f"text:{hashlib.md5(source[:100].encode()).hexdigest()[:8]}"
                sources_processed += 1

            chunks = chunk_text(content, self._chunk_size, self._chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    "text": chunk[:1000],  # Vectorize metadata limit
                    "source": source_label,
                    "chunk_index": str(i),
                }
                if metadata:
                    chunk_meta.update(metadata)
                all_chunks.append({"text": chunk, "metadata": chunk_meta})

        if not all_chunks:
            return {
                "chunks_created": 0,
                "vectors_stored": 0,
                "sources_processed": sources_processed,
            }

        # Embed all chunks
        texts = [c["text"] for c in all_chunks]
        embeddings = await self._embed(texts)

        # Build vectors for Vectorize upsert
        vectors = []
        for j, (chunk_data, embedding) in enumerate(zip(all_chunks, embeddings)):
            vec_id = hashlib.sha256(f"{chunk_data['metadata']['source']}:{j}".encode()).hexdigest()[
                :32
            ]
            vectors.append(
                {
                    "id": vec_id,
                    "values": embedding,
                    "metadata": chunk_data["metadata"],
                }
            )

        # Upsert in batches of 100
        stored = 0
        for i in range(0, len(vectors), 100):
            batch = vectors[i : i + 100]
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._vectorize_url("upsert"),
                    headers=self._headers,
                    json={"vectors": batch},
                )
                resp.raise_for_status()
            data = resp.json()
            check_api_response(data)
            stored += len(batch)

        return {
            "chunks_created": len(all_chunks),
            "vectors_stored": stored,
            "sources_processed": sources_processed,
        }

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        rerank: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the knowledge base with optional reranking.

        Pipeline: embed query → Vectorize search → (optional) rerank

        Args:
            query: Search query.
            top_k: Override default result count.
            rerank: Whether to rerank results.
            filters: Vectorize metadata filters.

        Returns:
            List of results with text, source, score.
        """
        k = top_k or (self._top_k if rerank else self._rerank_top_k)

        # Embed query
        embeddings = await self._embed([query])
        if not embeddings:
            return []

        # Vector search
        body: dict[str, Any] = {
            "vector": embeddings[0],
            "topK": k,
            "returnValues": False,
            "returnMetadata": "all",
        }
        if filters:
            body["filter"] = filters

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._vectorize_url("query"),
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()

        data = resp.json()
        check_api_response(data)
        matches = data.get("result", {}).get("matches", [])

        if not matches:
            return []

        # Build results
        results = []
        for m in matches:
            meta = m.get("metadata", {})
            results.append(
                {
                    "text": meta.get("text", ""),
                    "source": meta.get("source", ""),
                    "vector_score": m.get("score", 0.0),
                    "chunk_index": meta.get("chunk_index", ""),
                }
            )

        # Rerank
        if rerank and self._reranker_model and results:
            docs = [r["text"] for r in results]
            ranked = await self._rerank(query, docs)
            reranked = []
            for orig_idx, rerank_score in ranked[: self._rerank_top_k]:
                if orig_idx < len(results):
                    result = results[orig_idx].copy()
                    result["rerank_score"] = rerank_score
                    reranked.append(result)
            return reranked

        return results[: self._rerank_top_k]

    # Sync wrappers
    def ingest_sync(self, sources: list[str], **kwargs: Any) -> dict[str, Any]:
        import asyncio

        return asyncio.run(self.ingest(sources, **kwargs))

    def search_sync(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        import asyncio

        return asyncio.run(self.search(query, **kwargs))
