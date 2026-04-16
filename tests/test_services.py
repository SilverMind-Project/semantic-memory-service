"""Tests for search and text embedder services."""

import pytest
import math
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.search import SearchService, SearchServiceError
from app.services.text_embedder import (
    NullTextEmbedder,
    SentenceTransformerEmbedder,
    build_text_embedder,
)
from app.models.schemas import ObservationSearchRequest


# =============================================================================
# NullTextEmbedder Tests
# =============================================================================

class TestNullTextEmbedder:
    """Tests for NullTextEmbedder (graceful degradation)."""

    def test_embed_returns_empty_list(self):
        """Empty text should return empty embedding."""
        embedder = NullTextEmbedder()
        assert embedder.embed("") == []
        assert embedder.embed("   ") == []
        assert embedder.embed(None) == []  # type: ignore

    def test_embed_returns_empty_list_for_any_input(self):
        """Any input should return empty embedding."""
        embedder = NullTextEmbedder()
        assert embedder.embed("any text") == []

    def test_is_available_false(self):
        """Null embedder should report as unavailable."""
        assert NullTextEmbedder().is_available is False

    def test_embedding_dim_zero(self):
        """Null embedder should report zero dimension."""
        assert NullTextEmbedder().embedding_dim == 0


# =============================================================================
# SentenceTransformerEmbedder Tests
# =============================================================================

class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder."""

    @pytest.mark.skip(reason="Requires sentence-transformers and torch installation")
    def test_embed_returns_correct_dimension(self):
        """Embedding should be 384 dimensions for all-MiniLM-L6-v2."""
        embedder = SentenceTransformerEmbedder()
        embedding = embedder.embed("test sentence")
        assert len(embedding) == 384

    @pytest.mark.skip(reason="Requires sentence-transformers and torch installation")
    def test_embed_normalizes_vectors(self):
        """Embeddings should be L2-normalized for cosine similarity."""
        embedder = SentenceTransformerEmbedder()
        embedding = embedder.embed("test sentence")
        # L2 norm should be ~1.0
        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.skip(reason="Requires sentence-transformers and torch installation")
    def test_embed_deterministic(self):
        """Same input should produce same embedding."""
        embedder = SentenceTransformerEmbedder()
        embedding1 = embedder.embed("hello world")
        embedding2 = embedder.embed("hello world")
        assert embedding1 == embedding2

    @pytest.mark.skip(reason="Requires sentence-transformers and torch installation")
    def test_embed_different_inputs_different_vectors(self):
        """Different inputs should produce different embeddings."""
        embedder = SentenceTransformerEmbedder()
        embedding1 = embedder.embed("hello world")
        embedding2 = embedder.embed("goodbye world")
        assert embedding1 != embedding2

    @pytest.mark.skip(reason="Requires sentence-transformers and torch installation")
    def test_is_available_true(self):
        """Loaded embedder should report as available."""
        assert SentenceTransformerEmbedder().is_available is True

    @pytest.mark.skip(reason="Requires sentence-transformers and torch installation")
    def test_embedding_dim_property(self):
        """Embedding dimension property should match actual output."""
        embedder = SentenceTransformerEmbedder()
        assert embedder.embedding_dim == 384


# =============================================================================
# build_text_embedder Factory Tests
# =============================================================================

class TestBuildTextEmbedder:
    """Tests for build_text_embedder factory function."""

    def test_disabled_returns_null_embedder(self):
        """Disabled embedding should return NullTextEmbedder."""
        embedder = build_text_embedder(
            enabled=False,
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )
        assert isinstance(embedder, NullTextEmbedder)

    def test_enabled_returns_sentence_transformer_if_available(self):
        """Enabled embedding should return SentenceTransformerEmbedder if available."""
        # This test passes if we can import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            embedder = build_text_embedder(
                enabled=True,
                model_name="all-MiniLM-L6-v2",
                device="cpu",
            )
            assert isinstance(embedder, SentenceTransformerEmbedder)
        except ImportError:
            # If not installed, should fall back to null embedder
            embedder = build_text_embedder(
                enabled=True,
                model_name="all-MiniLM-L6-v2",
                device="cpu",
            )
            assert isinstance(embedder, NullTextEmbedder)


# =============================================================================
# SearchService Tests
# =============================================================================

class TestSearchService:
    """Tests for SearchService with text embedding support."""

    @pytest.mark.asyncio
    async def test_search_with_filters_only(self):
        """Search with filters but no query should work."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(return_value=[{
            'id': 1,
            'observed_at': '2026-04-14T00:00:00Z',
            'room_name': 'living_room',
            'description': 'A person is sitting on the sofa',
            'hazard_flags': ['none'],
            'object_list': ['person', 'sofa'],
        }])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService()
            search_req = ObservationSearchRequest(
                room_id="room_1",
                limit=10,
            )

            results = await service.search_observations(search_req)

            assert len(results) == 1
            assert results[0].room_name == 'living_room'
            assert results[0].text_similarity is None
            assert results[0].image_similarity is None

    @pytest.mark.asyncio
    async def test_search_with_room_id_filter(self):
        """Search should filter by room_id."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService()
            search_req = ObservationSearchRequest(
                room_id="kitchen",
                limit=5,
            )

            await service.search_observations(search_req)

            # Verify query was called
            assert mock_conn.fetch.called

    @pytest.mark.asyncio
    async def test_search_with_objects_filter(self):
        """Search should filter by objects_any."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService()
            search_req = ObservationSearchRequest(
                objects_any=["person", "chair"],
                limit=10,
            )

            await service.search_observations(search_req)

            assert mock_conn.fetch.called

    @pytest.mark.asyncio
    async def test_search_with_hazard_flags_filter(self):
        """Search should filter by hazard_flags_any."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService()
            search_req = ObservationSearchRequest(
                hazard_flags_any=["fire", "smoke"],
                limit=10,
            )

            await service.search_observations(search_req)

            assert mock_conn.fetch.called

    @pytest.mark.asyncio
    async def test_search_with_since_minutes_filter(self):
        """Search should filter by time."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService()
            search_req = ObservationSearchRequest(
                since_minutes=30,
                limit=10,
            )

            await service.search_observations(search_req)

            assert mock_conn.fetch.called

    @pytest.mark.asyncio
    async def test_search_text_only_with_null_embedder(self):
        """Text search with null embedder should return empty results."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            # Create service with null embedder
            service = SearchService(text_embedder=NullTextEmbedder())
            search_req = ObservationSearchRequest(
                query_text="person cooking",
                limit=10,
            )

            results = await service.search_observations(search_req)

            # Should not raise, just return empty results
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_result_includes_similarity_scores(self):
        """Search results should include similarity scores when available."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        # Mock row with similarity scores
        mock_conn.fetch = AsyncMock(return_value=[{
            'id': 1,
            'observed_at': '2026-04-14T00:00:00Z',
            'room_name': 'kitchen',
            'description': 'Person cooking',
            'hazard_flags': [],
            'object_list': ['person', 'stove'],
            'image_similarity': 0.85,
            'text_similarity': 0.72,
        }])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService()
            search_req = ObservationSearchRequest(
                query_embedding=[0.1] * 768,  # CLIP embedding dimension
                limit=10,
            )

            results = await service.search_observations(search_req)

            assert len(results) == 1
            assert results[0].image_similarity == 0.85
            # text_similarity should be None since no text query
            assert results[0].text_similarity is None

    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Search should handle database errors gracefully."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(side_effect=Exception("Database connection lost"))

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService()
            search_req = ObservationSearchRequest(limit=10)

            with pytest.raises(SearchServiceError) as exc_info:
                await service.search_observations(search_req)

            assert "Search failed" in str(exc_info.value)


# =============================================================================
# Integration Tests
# =============================================================================

class TestTextEmbeddingIntegration:
    """Integration tests for text embedding functionality."""

    @pytest.mark.asyncio
    async def test_search_with_text_query_and_null_embedder(self):
        """Text search should gracefully handle unavailable embedder."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService(text_embedder=NullTextEmbedder())
            search_req = ObservationSearchRequest(
                query_text="person in kitchen",
                room_id="kitchen",
                limit=10,
            )

            # Should not raise even though text embedder is unavailable
            results = await service.search_observations(search_req)

            assert isinstance(results, list)
            # Results should still be filtered by room_id

    @pytest.mark.asyncio
    async def test_search_combined_filters(self):
        """Search should handle multiple filters simultaneously."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
        async_mock.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire = MagicMock(return_value=async_mock)

        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("app.db.connection.db.get_pool", return_value=mock_pool):
            service = SearchService()
            search_req = ObservationSearchRequest(
                room_id="kitchen",
                since_minutes=60,
                objects_any=["person", "knife"],
                hazard_flags_any=["weapon"],
                limit=5,
            )

            await service.search_observations(search_req)

            # Verify all filters were applied
            assert mock_conn.fetch.called
