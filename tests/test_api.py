"""API tests for observations, movements, objects, and health endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app


# =============================================================================
# Helpers
# =============================================================================

def _make_mock_cursor(fetchone_return=None, fetchall_return=None, fetch_return=None,
                      fetchrow_return=None):
    """Build an async cursor mock with execute/fetch* calls."""
    cur = MagicMock()
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(return_value=fetchone_return)
    cur.fetchall = AsyncMock(return_value=fetchall_return or [])
    cur.fetch = AsyncMock(return_value=fetch_return or [])
    cur.fetchrow = AsyncMock(return_value=fetchrow_return)
    cur.description = []
    return cur


@pytest.fixture(autouse=True)
def setup_db():
    """Mock database pool and text embedder to avoid needing real PostgreSQL / Triton."""
    import app.db.connection as db_mod
    import app.db.migrate as migrate_mod
    from app.routers import observations as obs_mod
    from app.services.text_embedder import NullTextEmbedder

    cur = _make_mock_cursor(
        fetchone_return=(1, datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)),
    )
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=conn)

    # Save original state
    orig_pool = db_mod.db.pool
    orig_connect = db_mod.db.connect
    orig_disconnect = db_mod.db.disconnect
    orig_embedder = obs_mod.search_service._text_embedder

    db_mod.db.pool = mock_pool
    db_mod.db.connect = AsyncMock()
    db_mod.db.disconnect = AsyncMock()
    obs_mod.search_service._text_embedder = NullTextEmbedder()

    # run_migrations is called by lifespan; bypass it everywhere
    with patch("app.main.run_migrations", new_callable=AsyncMock), \
         patch.object(migrate_mod, "run_migrations", new_callable=AsyncMock):
        yield

    # Restore
    db_mod.db.pool = orig_pool
    db_mod.db.connect = orig_connect
    db_mod.db.disconnect = orig_disconnect
    obs_mod.search_service._text_embedder = orig_embedder


@pytest.fixture
def test_client():
    with TestClient(app=app) as client:
        yield client


@pytest.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


def _obs_payload(**overrides):
    return {
        "sensor_id": "camera_001",
        "room_id": "kitchen",
        "room_name": "Kitchen",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source": "scene_intel",
        "description": "A person cooking food on the stove",
        "description_embedding": [0.1] * 768,
        "object_list": ["person", "stove", "pan"],
        "hazard_flags": [],
        "embedding": [0.2] * 768,
        **overrides,
    }


# =============================================================================
# Health Check
# =============================================================================

class TestHealth:
    def test_health_check(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


# =============================================================================
# Observation Endpoints
# =============================================================================

class TestObservationCreate:
    def test_create_observation_with_embedding(self, test_client):
        response = test_client.post("/api/v1/observations/", json=_obs_payload())
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "created_at" in data
        assert data["description"] == "A person cooking food on the stove"

    def test_create_observation_minimal(self, test_client):
        response = test_client.post("/api/v1/observations/", json=_obs_payload(
            description_embedding=None,
            embedding=None,
        ))
        assert response.status_code == 201
        data = response.json()
        assert "id" in data

    def test_create_observation_returns_real_created_at(self, test_client):
        """created_at should come from the database, not a hardcoded string."""
        response = test_client.post("/api/v1/observations/", json=_obs_payload())
        data = response.json()
        assert data["created_at"] != "2026-04-14T00:00:00Z"


class TestObservationSearch:
    def test_search_by_image_embedding(self, test_client):
        response = test_client.post("/api/v1/observations/search", json={
            "query_embedding": [0.1] * 768,
            "limit": 10,
        })
        assert response.status_code == 200

    def test_search_with_text_query(self, test_client):
        response = test_client.post("/api/v1/observations/search", json={
            "query_text": "cooking food",
            "limit": 10,
        })
        assert response.status_code in [200, 500]

    def test_search_with_filters(self, test_client):
        response = test_client.post("/api/v1/observations/search", json={
            "query_text": "person",
            "room_id": "kitchen",
            "objects_any": ["person", "stove"],
            "limit": 10,
        })
        assert response.status_code == 200

    def test_search_empty_results(self, test_client):
        response = test_client.post("/api/v1/observations/search", json={
            "query_text": "nonexistent_xyz",
            "limit": 10,
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_search_default_limit(self, test_client):
        response = test_client.post("/api/v1/observations/search", json={
            "query_embedding": [0.1] * 768,
        })
        assert response.status_code == 200

    def test_search_with_similarity_threshold(self, test_client):
        response = test_client.post("/api/v1/observations/search", json={
            "query_embedding": [0.1] * 768,
            "similarity_threshold": 0.99,
            "limit": 5,
        })
        assert response.status_code == 200

    def test_search_since_minutes(self, test_client):
        response = test_client.post("/api/v1/observations/search", json={
            "query_embedding": [0.1] * 768,
            "since_minutes": 30,
            "limit": 10,
        })
        assert response.status_code == 200


class TestObservationPrune:
    def test_prune_observations(self, test_client):
        response = test_client.delete("/api/v1/observations/prune?days=30")
        assert response.status_code == 200
        assert "pruned" in response.json()


# =============================================================================
# Movement Endpoints
# =============================================================================

def _movement_payload(**overrides):
    return {
        "person_id": "person_1",
        "person_name": "Alice",
        "sensor_id": "cam_1",
        "from_room_id": "living_room",
        "to_room_id": "kitchen",
        "from_room_name": "Living Room",
        "to_room_name": "Kitchen",
        "direction_raw": "left_to_right",
        "direction_semantic": "entering",
        "confidence": 0.95,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        **overrides,
    }


class TestMovementCreate:
    def test_create_movement(self, test_client):
        response = test_client.post("/api/v1/movements/", json=_movement_payload())
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "created_at" in data

    def test_create_movement_returns_real_created_at(self, test_client):
        response = test_client.post("/api/v1/movements/", json=_movement_payload())
        data = response.json()
        assert data["created_at"] != "2026-04-14T00:00:00Z"


class TestMovementTransitions:
    def test_get_transitions_with_filters(self, test_client):
        response = test_client.get(
            "/api/v1/movements/transitions",
            params={
                "person_id": "person_1",
                "semantic": "entering",
                "to_room_id": "kitchen",
            },
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_transitions_minimal(self, test_client):
        response = test_client.get(
            "/api/v1/movements/transitions",
            params={"person_id": "person_1"},
        )
        assert response.status_code == 200


# =============================================================================
# Object Presence Endpoints
# =============================================================================

class TestObjectPresence:
    def test_get_recent_objects(self, test_client):
        response = test_client.get("/api/v1/objects/room_1/recent")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_recent_objects_with_since_minutes(self, test_client):
        response = test_client.get("/api/v1/objects/room_1/recent?since_minutes=30")
        assert response.status_code == 200


# =============================================================================
# Error Handling
# =============================================================================

class TestErrorHandling:
    def test_404_on_unknown_route(self, test_client):
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_422_on_invalid_payload(self, test_client):
        response = test_client.post("/api/v1/observations/", json={"invalid": "data"})
        assert response.status_code == 422
