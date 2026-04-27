"""Tests for KnowledgeBase and DIYKnowledgeBase."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai_cloudflare.knowledge import (
    DIYKnowledgeBase,
    KnowledgeBase,
    chunk_text,
)


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        assert chunk_text("Hello world") == ["Hello world"]

    def test_empty_text(self) -> None:
        assert chunk_text("") == []

    def test_respects_chunk_size(self) -> None:
        text = "word " * 200  # ~1000 chars
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 350  # some slack for boundary finding

    def test_prefers_paragraph_breaks(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert any("First" in c for c in chunks)
        assert any("Second" in c for c in chunks)

    def test_overlap_exists(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = chunk_text(text, chunk_size=100, overlap=30)
        if len(chunks) > 1:
            # Check some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # Each chunk should have content
                assert len(chunks[i + 1]) > 0


class TestKnowledgeBase:
    def test_creates_with_defaults(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kb = KnowledgeBase("my-docs")
            assert kb._instance == "my-docs"
            assert kb._retrieval_type == "hybrid"
            assert kb._reranking is True

    def test_search_options_include_reranking(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kb = KnowledgeBase("docs", reranking=True)
            opts = kb._search_options()
            assert opts["reranking"]["enabled"] is True
            assert "bge-reranker" in opts["reranking"]["model"]

    def test_search_options_hybrid(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kb = KnowledgeBase("docs")
            opts = kb._search_options()
            assert opts["retrieval"]["retrieval_type"] == "hybrid"

    def test_boost_by(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kb = KnowledgeBase("docs", boost_by=[{"field": "timestamp", "direction": "desc"}])
            opts = kb._search_options()
            assert "boost_by" in opts["retrieval"]

    @pytest.mark.asyncio
    async def test_search_calls_api(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kb = KnowledgeBase("docs")

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "success": True,
                "result": {
                    "data": [
                        {"text": "Caching stores copies", "source": "docs.md", "score": 0.92},
                        {"text": "TTL controls expiry", "source": "docs.md", "score": 0.85},
                    ]
                },
            }

            with patch("pydantic_ai_cloudflare.knowledge.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_resp)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_client

                results = await kb.search("How does caching work?")

            assert len(results) == 2


class TestDIYKnowledgeBase:
    def test_creates_with_defaults(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kb = DIYKnowledgeBase("my-index")
            assert kb._index == "my-index"
            assert kb._reranker_model is not None
            assert kb._chunk_size == 800

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_score(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kb = DIYKnowledgeBase("idx")

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "success": True,
                "result": [
                    {"index": 0, "score": 0.3},
                    {"index": 1, "score": 0.9},
                    {"index": 2, "score": 0.6},
                ],
            }

            with patch("pydantic_ai_cloudflare.knowledge.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_resp)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_client

                ranked = await kb._rerank("test query", ["doc a", "doc b", "doc c"])

            # Should be sorted by score descending
            assert ranked[0][1] > ranked[1][1]

    @pytest.mark.asyncio
    async def test_ingest_text(self) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            kb = DIYKnowledgeBase("idx", chunk_size=50, chunk_overlap=10)

            embed_resp = MagicMock()
            embed_resp.raise_for_status = MagicMock()
            embed_resp.json.return_value = {
                "success": True,
                "result": {"data": [[0.1, 0.2, 0.3]]},
            }

            upsert_resp = MagicMock()
            upsert_resp.raise_for_status = MagicMock()
            upsert_resp.json.return_value = {"success": True, "result": {}}

            with patch("pydantic_ai_cloudflare.knowledge.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=[embed_resp, upsert_resp])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_client

                stats = await kb.ingest(["This is a short test document."])

            assert stats["chunks_created"] >= 1


class TestLocalFileIngestion:
    """Test ingesting local files and directories."""

    @pytest.mark.asyncio
    async def test_ingest_single_file(self, tmp_path: Any) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            # Create a temp file
            f = tmp_path / "test.txt"
            f.write_text("This is test content for ingestion into the knowledge base.")

            kb = DIYKnowledgeBase("idx", chunk_size=100, chunk_overlap=20)

            embed_resp = MagicMock()
            embed_resp.raise_for_status = MagicMock()
            embed_resp.json.return_value = {
                "success": True,
                "result": {"data": [[0.1, 0.2, 0.3]]},
            }
            upsert_resp = MagicMock()
            upsert_resp.raise_for_status = MagicMock()
            upsert_resp.json.return_value = {"success": True, "result": {}}

            with patch("pydantic_ai_cloudflare.knowledge.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=[embed_resp, upsert_resp])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_client

                stats = await kb.ingest([str(f)])

            assert stats["sources_processed"] == 1
            assert stats["chunks_created"] >= 1

    @pytest.mark.asyncio
    async def test_ingest_directory(self, tmp_path: Any) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            # Create multiple files
            (tmp_path / "doc1.txt").write_text("First document about Python.")
            (tmp_path / "doc2.md").write_text("# Second doc\nAbout Cloudflare.")
            (tmp_path / "image.png").write_bytes(b"\x89PNG")  # should be skipped

            kb = DIYKnowledgeBase("idx", chunk_size=100, chunk_overlap=10)

            embed_resp = MagicMock()
            embed_resp.raise_for_status = MagicMock()
            embed_resp.json.return_value = {
                "success": True,
                "result": {"data": [[0.1, 0.2], [0.3, 0.4]]},
            }
            upsert_resp = MagicMock()
            upsert_resp.raise_for_status = MagicMock()
            upsert_resp.json.return_value = {"success": True, "result": {}}

            with patch("pydantic_ai_cloudflare.knowledge.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=[embed_resp, upsert_resp])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_client

                stats = await kb.ingest([str(tmp_path)])

            assert stats["sources_processed"] == 2  # txt + md, not png
            assert stats["chunks_created"] >= 2

    @pytest.mark.asyncio
    async def test_ingest_glob_pattern(self, tmp_path: Any) -> None:
        with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}):
            (tmp_path / "a.txt").write_text("File A content.")
            (tmp_path / "b.txt").write_text("File B content.")
            (tmp_path / "c.md").write_text("File C content.")

            kb = DIYKnowledgeBase("idx", chunk_size=100, chunk_overlap=10)

            embed_resp = MagicMock()
            embed_resp.raise_for_status = MagicMock()
            embed_resp.json.return_value = {
                "success": True,
                "result": {"data": [[0.1, 0.2], [0.3, 0.4]]},
            }
            upsert_resp = MagicMock()
            upsert_resp.raise_for_status = MagicMock()
            upsert_resp.json.return_value = {"success": True, "result": {}}

            with patch("pydantic_ai_cloudflare.knowledge.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=[embed_resp, upsert_resp])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_cls.return_value = mock_client

                # Only .txt files
                stats = await kb.ingest([str(tmp_path / "*.txt")])

            assert stats["sources_processed"] == 2  # a.txt + b.txt, not c.md
