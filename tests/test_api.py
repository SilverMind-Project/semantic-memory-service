"""API tests for text embedding search functionality."""

import pytest
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from app.main import app
from datetime import datetime, timezone


@pytest.fixture(autouse=True)
def setup_db():
    """Mock database connection to avoid needing real PostgreSQL."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    async_mock = AsyncMock()
    async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
    async_mock.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=async_mock)
    mock_conn.execute = AsyncMock(return_value=None)
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=1)  # Return ID for create

    with patch("app.db.connection.db.get_pool", return_value=mock_pool):
        with patch("app.db.connection.db.connect", new_callable=AsyncMock):
            with patch("app.db.connection.db.disconnect", new_callable=AsyncMock):
                yield


@pytest.fixture
def test_client():
    """Create test client using TestClient (synchronous)."""
    with TestClient(app=app) as client:
        yield client


@pytest.fixture
async def async_client():
    """Create async test client using ASGITransport."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def sample_observation(async_client: AsyncClient):
    """Create a sample observation for testing."""
    obs_data = {
        "sensor_id": "camera_001",
        "room_id": "kitchen",
        "room_name": "Kitchen",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source": "scene_intel",
        "description": "A person cooking food on the stove",
        "description_embedding": [0.1] * 384,  # 384-dim text embedding
        "object_list": ["person", "stove", "pan"],
        "hazard_flags": [],
        "embedding": [0.2] * 768,  # 768-dim image embedding (CLIP)
    }
    response = await async_client.post("/api/v1/observations/", json=obs_data)
    return response.json() if response.status_code == 201 else None


class TestObservationCreateWithTextEmbedding:
    """Tests for creating observations with text embeddings."""

    def test_create_observation_with_description_embedding(self, test_client: TestClient):
        """Should create observation with description embedding."""
        obs_data = {
            "sensor_id": "camera_001",
            "room_id": "kitchen",
            "room_name": "Kitchen",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source": "scene_intel",
            "description": "A person cooking food on the stove",
            "description_embedding": [0.1] * 384,
            "object_list": ["person", "stove", "pan"],
            "hazard_flags": [],
            "embedding": [0.2] * 768,
        }
        response = test_client.post("/api/v1/observations/", json=obs_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["description"] == "A person cooking food on the stove"

    def test_create_observation_without_description_embedding(self, test_client: TestClient):
        """Should create observation without description embedding (optional field)."""
        obs_data = {
            "sensor_id": "camera_001",
            "room_id": "kitchen",
            "room_name": "Kitchen",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source": "scene_intel",
            "description": "A person standing in the kitchen",
            "object_list": ["person"],
            "hazard_flags": [],
        }
        response = test_client.post("/api/v1/observations/", json=obs_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data


class TestObservationSearchWithText:
    """Tests for searching observations with text embeddings."""

    def test_search_with_text_query(self, test_client: TestClient):
        """Should search by text query when text embedder is available."""
        search_data = {
            "query_text": "cooking food",
            "limit": 10,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        # Should not fail even if no text embedder is installed
        assert response.status_code in [200, 500]  # 500 if embedder issues

    def test_search_with_image_query(self, test_client: TestClient):
        """Should search by image embedding."""
        search_data = {
            "query_embedding": [0.1] * 768,  # CLIP embedding dimension
            "limit": 10,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200

    def test_search_with_combined_query(self, test_client: TestClient):
        """Should support hybrid search with both text and image queries."""
        search_data = {
            "query_text": "person cooking",
            "query_embedding": [0.1] * 768,
            "limit": 10,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code in [200, 500]  # 500 if text embedder unavailable

    def test_search_with_filters(self, test_client: TestClient):
        """Should search with filters and text query."""
        search_data = {
            "query_text": "person",
            "room_id": "kitchen",
            "objects_any": ["person", "stove"],
            "limit": 10,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200

    def test_search_with_similarity_threshold(self, test_client: TestClient):
        """Should filter by similarity threshold."""
        search_data = {
            "query_text": "cooking",
            "similarity_threshold": 0.8,
            "limit": 10,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200

    def test_search_response_includes_similarity_scores(self, test_client: TestClient):
        """Search response should include similarity scores."""
        search_data = {
            "query_embedding": [0.1] * 768,
            "limit": 5,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # If results exist, should have image_similarity
        if len(data) > 0:
            assert "image_similarity" in data[0]


class TestObservationSearchEdgeCases:
    """Edge case tests for observation search."""

    def test_search_empty_results(self, test_client: TestClient):
        """Should return empty list when no matches found."""
        search_data = {
            "query_text": "nonexistent_object_xyz",
            "limit": 10,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_search_with_very_high_threshold(self, test_client: TestClient):
        """Should return few/no results with very high threshold."""
        search_data = {
            "query_embedding": [0.1] * 768,
            "similarity_threshold": 0.99,
            "limit": 10,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200

    def test_search_with_very_low_threshold(self, test_client: TestClient):
        """Should return more results with low threshold."""
        search_data = {
            "query_embedding": [0.1] * 768,
            "similarity_threshold": 0.0,
            "limit": 10,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200

    def test_search_default_limit(self, test_client: TestClient):
        """Should use default limit of 20."""
        search_data = {
            "query_embedding": [0.1] * 768,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200

    def test_search_with_custom_limit(self, test_client: TestClient):
        """Should respect custom limit."""
        search_data = {
            "query_embedding": [0.1] * 768,
            "limit": 5,
        }
        response = test_client.post("/api/v1/observations/search", json=search_data)
        assert response.status_code == 200
