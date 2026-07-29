"""Tests for services: text embedder, search, observation store, movement store, object presence."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest

from app.models.schemas import (
    ObservationCreate,
    ObservationSearchRequest,
    MovementCreate,
)
from app.services.search import SearchService, SearchServiceError
from app.services.observation_store import ObservationStore, ObservationStoreError
from app.services.movement_store import MovementStore, MovementStoreError
from app.services.object_presence import ObjectPresenceStore, ObjectPresenceStoreError
from app.services.text_embedder import (
    NullTextEmbedder,
    TritonTextEmbedder,
    build_text_embedder,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_mock_cursor(return_value=None, fetchall_return=None, fetchone_return=None,
                      description=None):
    """Build a mock async cursor with execute, fetch, fetchone, fetchall."""
    cur = MagicMock()
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(return_value=fetchone_return)
    cur.fetchall = AsyncMock(return_value=fetchall_return or [])
    cur.fetch = AsyncMock(return_value=return_value or [])
    cur.description = description or []
    return cur


def _make_mock_conn(cursor):
    """Build a mock connection whose .cursor() yields the given cursor."""
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    return conn


def _mock_db_pool(monkeypatch, cursor):
    """Patch db.get_pool so pool.connection() → conn → conn.cursor() → cursor."""
    conn = _make_mock_conn(cursor)
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=conn)
    monkeypatch.setattr("app.db.connection.db.get_pool", lambda: mock_pool)
    return mock_pool


# =============================================================================
# NullTextEmbedder Tests
# =============================================================================

class TestNullTextEmbedder:
    @pytest.mark.asyncio
    async def test_embed_returns_empty_list_for_any_input(self):
        embedder = NullTextEmbedder()
        assert await embedder.embed("") == []
        assert await embedder.embed("   ") == []
        assert await embedder.embed("any text") == []

    def test_is_available_false(self):
        assert NullTextEmbedder().is_available is False

    def test_embedding_dim_zero(self):
        assert NullTextEmbedder().embedding_dim == 0


# =============================================================================
# TritonTextEmbedder Tests
# =============================================================================

class TestTritonTextEmbedder:
    def test_embedding_dim_property(self):
        embedder = TritonTextEmbedder(
            triton_url="localhost:8701",
            model_name="embeddinggemma-300m",
            tokenizer_path="/tmp/tokenizer.json",
        )
        assert embedder.embedding_dim == 768

    def test_is_available_true(self):
        embedder = TritonTextEmbedder(
            triton_url="localhost:8701",
            model_name="embeddinggemma-300m",
            tokenizer_path="/tmp/tokenizer.json",
        )
        assert embedder.is_available is True

    @pytest.mark.asyncio
    async def test_embed_empty_text_returns_empty_list(self):
        embedder = TritonTextEmbedder(
            triton_url="localhost:8701",
            model_name="embeddinggemma-300m",
            tokenizer_path="/tmp/tokenizer.json",
        )
        assert await embedder.embed("") == []
        assert await embedder.embed("   ") == []


# =============================================================================
# build_text_embedder Factory Tests
# =============================================================================

class TestBuildTextEmbedder:
    def test_disabled_returns_null_embedder(self):
        embedder = build_text_embedder(
            enabled=False,
            triton_url="localhost:8701",
            model_name="embeddinggemma-300m",
            tokenizer_path="/tmp/tokenizer.json",
        )
        assert isinstance(embedder, NullTextEmbedder)

    def test_enabled_returns_triton_embedder(self):
        embedder = build_text_embedder(
            enabled=True,
            triton_url="localhost:8701",
            model_name="embeddinggemma-300m",
            tokenizer_path="/tmp/tokenizer.json",
        )
        assert isinstance(embedder, TritonTextEmbedder)
        assert embedder.embedding_dim == 768


# =============================================================================
# SearchService Tests
# =============================================================================

class TestSearchService:

    @pytest.mark.asyncio
    async def test_search_with_filters_only(self, monkeypatch):
        cursor = _make_mock_cursor(
            fetchall_return=[(
                1, "2026-04-14T00:00:00Z", "room_1", "living_room",
                "A person is sitting on the sofa", ["none"], ["person", "sofa"],
            )],
            description=[
                ("id",), ("observed_at",), ("room_id",), ("room_name",),
                ("description",), ("hazard_flags",), ("object_list",),
            ],
        )
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        results = await service.search_observations(
            ObservationSearchRequest(room_id="room_1", limit=10)
        )

        assert len(results) == 1
        assert results[0].room_name == "living_room"
        assert results[0].text_similarity is None
        assert results[0].image_similarity is None

    @pytest.mark.asyncio
    async def test_search_with_room_id_filter(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(
            ObservationSearchRequest(room_id="kitchen", limit=5)
        )
        assert cursor.execute.called

    @pytest.mark.asyncio
    async def test_search_with_objects_filter(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(
            ObservationSearchRequest(objects_any=["person", "chair"], limit=10)
        )
        assert cursor.execute.called

    @pytest.mark.asyncio
    async def test_search_with_hazard_flags_filter(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(
            ObservationSearchRequest(hazard_flags_any=["fire", "smoke"], limit=10)
        )
        assert cursor.execute.called

    @pytest.mark.asyncio
    async def test_search_with_since_minutes_filter(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(
            ObservationSearchRequest(since_minutes=30, limit=10)
        )
        assert cursor.execute.called

    @pytest.mark.asyncio
    async def test_search_text_only_with_null_embedder(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService(text_embedder=NullTextEmbedder())
        results = await service.search_observations(
            ObservationSearchRequest(query_text="person cooking", limit=10)
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_result_includes_similarity_scores(self, monkeypatch):
        cursor = _make_mock_cursor(
            fetchall_return=[(
                1, "2026-04-14T00:00:00Z", "room_1", "kitchen", "Person cooking",
                [], ["person", "stove"], 0.85,
            )],
            description=[
                ("id",), ("observed_at",), ("room_id",), ("room_name",),
                ("description",), ("hazard_flags",), ("object_list",),
                ("image_similarity",),
            ],
        )
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        results = await service.search_observations(
            ObservationSearchRequest(query_embedding=[0.1] * 768, limit=10)
        )

        assert len(results) == 1
        assert results[0].image_similarity == 0.85
        assert results[0].text_similarity is None

    @pytest.mark.asyncio
    async def test_search_with_text_query_and_null_embedder(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService(text_embedder=NullTextEmbedder())
        results = await service.search_observations(
            ObservationSearchRequest(
                query_text="person in kitchen", room_id="kitchen", limit=10
            )
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_combined_filters(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(
            ObservationSearchRequest(
                room_id="kitchen",
                since_minutes=60,
                objects_any=["person", "knife"],
                hazard_flags_any=["weapon"],
                limit=5,
            )
        )
        assert cursor.execute.called

    @pytest.mark.asyncio
    async def test_search_error_handling(self, monkeypatch):
        cursor = _make_mock_cursor()
        cursor.execute = AsyncMock(side_effect=Exception("Database connection lost"))
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        with pytest.raises(SearchServiceError, match="Search failed"):
            await service.search_observations(ObservationSearchRequest(limit=10))

    @pytest.mark.asyncio
    async def test_search_with_person_id_filter(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(
            ObservationSearchRequest(person_id="amma", limit=10)
        )
        sql = cursor.execute.call_args[0][0]
        params = cursor.execute.call_args[0][1]
        assert "person_id = %s" in sql
        assert "amma" in params

    @pytest.mark.asyncio
    async def test_search_with_kind_scene_matches_legacy_null(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(
            ObservationSearchRequest(kind="scene", limit=10)
        )
        sql = cursor.execute.call_args[0][0]
        assert "(kind = %s OR kind IS NULL)" in sql

    @pytest.mark.asyncio
    async def test_search_with_kind_guided_episode_is_exact_match(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(
            ObservationSearchRequest(kind="guided_episode", limit=10)
        )
        sql = cursor.execute.call_args[0][0]
        assert "kind = %s" in sql
        assert "(kind = %s OR kind IS NULL)" not in sql

    @pytest.mark.asyncio
    async def test_search_result_includes_person_id_and_kind(self, monkeypatch):
        cursor = _make_mock_cursor(
            fetchall_return=[(
                1, "2026-04-14T00:00:00Z", "room_1", "living_room",
                "Guided routine ended", [], ["tea"], "amma", "guided_episode",
            )],
            description=[
                ("id",), ("observed_at",), ("room_id",), ("room_name",),
                ("description",), ("hazard_flags",), ("object_list",),
                ("person_id",), ("kind",),
            ],
        )
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        results = await service.search_observations(
            ObservationSearchRequest(kind="guided_episode", limit=10)
        )
        assert results[0].person_id == "amma"
        assert results[0].kind == "guided_episode"

    @pytest.mark.asyncio
    async def test_search_unfiltered_unchanged(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        service = SearchService()
        await service.search_observations(ObservationSearchRequest(limit=10))
        sql = cursor.execute.call_args[0][0]
        assert "person_id = %s" not in sql
        assert "kind = %s" not in sql
        assert "kind IS NULL" not in sql


# =============================================================================
# ObservationStore Tests
# =============================================================================

class TestObservationStore:

    @pytest.mark.asyncio
    async def test_create_returns_id_and_created_at(self, monkeypatch):
        cursor = _make_mock_cursor(fetchone_return=(42, datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)))
        _mock_db_pool(monkeypatch, cursor)

        store = ObservationStore()
        obs = ObservationCreate(
            observed_at=datetime.now(timezone.utc),
            source="scene_intel",
        )
        obs_id, created_at = await store.create(obs)
        assert obs_id == 42
        assert created_at.year == 2026 and created_at.month == 5 and created_at.day == 7

    @pytest.mark.asyncio
    async def test_create_also_records_object_presence(self, monkeypatch):
        """object_presence must be written on the observation's own cursor.

        Nothing populated that table before, so get_recent_objects always
        returned [] and every room-context summary lost its object list.
        """
        observed = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
        cursor = _make_mock_cursor(fetchone_return=(42, observed))
        _mock_db_pool(monkeypatch, cursor)

        await ObservationStore().create(
            ObservationCreate(
                observed_at=observed,
                source="scene_intel",
                room_id="kitchen",
                object_list=["cup", "person"],
            )
        )

        presence_calls = [
            c for c in cursor.execute.await_args_list if "object_presence" in c.args[0]
        ]
        assert len(presence_calls) == 2, "one upsert per distinct label"
        # Sorted for determinism, and carrying the observation's capture time
        # rather than NOW().
        assert presence_calls[0].args[1] == ("kitchen", "cup", observed, observed, 42)
        assert presence_calls[1].args[1] == ("kitchen", "person", observed, observed, 42)

    @pytest.mark.asyncio
    async def test_create_skips_presence_without_a_room(self, monkeypatch):
        """Presence is keyed by room; a guided episode has none."""
        observed = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
        cursor = _make_mock_cursor(fetchone_return=(42, observed))
        _mock_db_pool(monkeypatch, cursor)

        await ObservationStore().create(
            ObservationCreate(
                observed_at=observed,
                source="guided_companion",
                room_id=None,
                object_list=["morning_routine"],
            )
        )

        assert not [
            c for c in cursor.execute.await_args_list if "object_presence" in c.args[0]
        ]

    @pytest.mark.asyncio
    async def test_create_raises_on_failure(self, monkeypatch):
        cursor = _make_mock_cursor(fetchone_return=None)
        _mock_db_pool(monkeypatch, cursor)

        store = ObservationStore()
        obs = ObservationCreate(
            observed_at=datetime.now(timezone.utc),
            source="scene_intel",
        )
        with pytest.raises(ObservationStoreError, match="Failed to create observation"):
            await store.create(obs)

    @pytest.mark.asyncio
    async def test_create_raises_on_db_error(self, monkeypatch):
        cursor = _make_mock_cursor()
        cursor.execute = AsyncMock(side_effect=OSError("connection lost"))
        _mock_db_pool(monkeypatch, cursor)

        store = ObservationStore()
        obs = ObservationCreate(
            observed_at=datetime.now(timezone.utc),
            source="scene_intel",
        )
        with pytest.raises(ObservationStoreError, match="Failed to create observation"):
            await store.create(obs)

    @pytest.mark.asyncio
    async def test_create_passes_person_id_and_kind(self, monkeypatch):
        cursor = _make_mock_cursor(fetchone_return=(7, datetime.now(timezone.utc)))
        _mock_db_pool(monkeypatch, cursor)

        store = ObservationStore()
        obs = ObservationCreate(
            room_id=None,
            observed_at=datetime.now(timezone.utc),
            source="guided_companion",
            person_id="amma",
            kind="guided_episode",
        )
        await store.create(obs)
        params = cursor.execute.call_args[0][1]
        assert params[-2] == "amma"
        assert params[-1] == "guided_episode"

    @pytest.mark.asyncio
    async def test_create_defaults_person_id_and_kind_to_none(self, monkeypatch):
        cursor = _make_mock_cursor(fetchone_return=(8, datetime.now(timezone.utc)))
        _mock_db_pool(monkeypatch, cursor)

        store = ObservationStore()
        obs = ObservationCreate(
            observed_at=datetime.now(timezone.utc),
            source="scene_intel",
        )
        await store.create(obs)
        params = cursor.execute.call_args[0][1]
        assert params[-2] is None
        assert params[-1] is None


# =============================================================================
# MovementStore Tests
# =============================================================================

class TestMovementStore:

    @pytest.mark.asyncio
    async def test_create_returns_id_and_created_at(self, monkeypatch):
        cursor = _make_mock_cursor(fetchone_return=(7, datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)))
        _mock_db_pool(monkeypatch, cursor)

        store = MovementStore()
        m = MovementCreate(
            person_id="p1",
            direction_raw="left_to_right",
            direction_semantic="entering",
            confidence=0.9,
            observed_at=datetime.now(timezone.utc),
        )
        m_id, created_at = await store.create(m)
        assert m_id == 7
        assert created_at.year == 2026 and created_at.month == 5 and created_at.day == 7

    @pytest.mark.asyncio
    async def test_create_raises_on_failure(self, monkeypatch):
        cursor = _make_mock_cursor(fetchone_return=None)
        _mock_db_pool(monkeypatch, cursor)

        store = MovementStore()
        m = MovementCreate(
            person_id="p1",
            direction_raw="left_to_right",
            direction_semantic="entering",
            confidence=0.9,
            observed_at=datetime.now(timezone.utc),
        )
        with pytest.raises(MovementStoreError, match="Failed to create movement"):
            await store.create(m)

    @pytest.mark.asyncio
    async def test_get_transitions(self, monkeypatch):
        cursor = _make_mock_cursor(
            fetchall_return=[(
                1, "p1", "Alice", "living_room", "kitchen",
                "Living Room", "Kitchen", "entering", 0.95,
                datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc), None,
            )],
            description=[
                ("id",), ("person_id",), ("person_name",), ("from_room_id",),
                ("to_room_id",), ("from_room_name",), ("to_room_name",),
                ("direction_semantic",), ("confidence",), ("observed_at",),
                ("observation_id",),
            ],
        )
        _mock_db_pool(monkeypatch, cursor)

        store = MovementStore()
        results = await store.get_transitions("p1")
        assert len(results) == 1
        assert results[0].person_name == "Alice"
        assert results[0].direction_semantic == "entering"

    @pytest.mark.asyncio
    async def test_get_transitions_with_filters(self, monkeypatch):
        cursor = _make_mock_cursor(fetchall_return=[])
        _mock_db_pool(monkeypatch, cursor)

        store = MovementStore()
        results = await store.get_transitions(
            "p1", semantic="entering", to_room_id="kitchen", since_minutes=60
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_get_transitions_raises_on_db_error(self, monkeypatch):
        cursor = _make_mock_cursor()
        cursor.execute = AsyncMock(side_effect=OSError("connection lost"))
        _mock_db_pool(monkeypatch, cursor)

        store = MovementStore()
        with pytest.raises(MovementStoreError, match="Failed to get transitions"):
            await store.get_transitions("p1")


# =============================================================================
# ObjectPresenceStore Tests
# =============================================================================

class TestObjectPresenceStore:

    @pytest.mark.asyncio
    async def test_upsert_presence(self, monkeypatch):
        cursor = _make_mock_cursor()
        _mock_db_pool(monkeypatch, cursor)

        store = ObjectPresenceStore()
        await store.upsert_presence("room_1", "chair", 123)
        assert cursor.execute.called

    @pytest.mark.asyncio
    async def test_get_by_room(self, monkeypatch):
        cursor = _make_mock_cursor(
            fetchall_return=[(
                "chair", datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc), 5,
            )],
            description=[
                ("object_label",), ("last_seen_at",), ("observation_count",),
            ],
        )
        _mock_db_pool(monkeypatch, cursor)

        store = ObjectPresenceStore()
        results = await store.get_by_room("room_1", 60)
        assert len(results) == 1
        assert results[0]["object_label"] == "chair"
        assert results[0]["observation_count"] == 5

    @pytest.mark.asyncio
    async def test_delete_old_records(self, monkeypatch):
        cursor = _make_mock_cursor(fetchall_return=[(1,), (2,)])
        _mock_db_pool(monkeypatch, cursor)

        store = ObjectPresenceStore()
        deleted = await store.delete_old_records(90)
        assert deleted == 2

    @pytest.mark.asyncio
    async def test_upsert_raises_on_db_error(self, monkeypatch):
        cursor = _make_mock_cursor()
        cursor.execute = AsyncMock(side_effect=OSError("connection lost"))
        _mock_db_pool(monkeypatch, cursor)

        store = ObjectPresenceStore()
        with pytest.raises(ObjectPresenceStoreError, match="Failed to upsert presence"):
            await store.upsert_presence("room_1", "chair", 123)
