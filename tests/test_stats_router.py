"""Router tests for write-health stats (DL-M01)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _make_mock_cursor(fetchone_side_effect, fetchall_return):
    cur = MagicMock()
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(side_effect=fetchone_side_effect)
    cur.fetchall = AsyncMock(return_value=fetchall_return)
    cur.description = []
    return cur


def _install_pool(cursor):
    import app.db.connection as db_mod

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=conn)

    orig_pool = db_mod.db.pool
    db_mod.db.pool = mock_pool
    return orig_pool


def _restore_pool(orig_pool):
    import app.db.connection as db_mod

    db_mod.db.pool = orig_pool


@pytest.fixture
def test_client():
    import app.db.migrate as migrate_mod

    with patch("app.main.run_migrations", new_callable=AsyncMock), \
         patch.object(migrate_mod, "run_migrations", new_callable=AsyncMock):
        with TestClient(app=app) as client:
            yield client


class TestWriteHealthEmpty:
    def test_write_health_with_empty_db_returns_nulls_and_zeros(self, test_client):
        cur = _make_mock_cursor(
            fetchone_side_effect=[(None,), (None,), (0,), (0,)],
            fetchall_return=[],
        )
        orig_pool = _install_pool(cur)
        try:
            response = test_client.get("/api/v1/stats/write-health")
        finally:
            _restore_pool(orig_pool)

        assert response.status_code == 200
        data = response.json()
        assert data["last_observation_at"] is None
        assert data["last_movement_at"] is None
        assert data["total_observations"] == 0
        assert data["total_movements"] == 0
        assert data["observations_by_day"] == []


class TestWriteHealthSeeded:
    def test_write_health_with_seeded_data_groups_by_day_and_source(self, test_client):
        last_obs = datetime(2026, 7, 21, 14, 0, 0, tzinfo=timezone.utc)
        last_mov = datetime(2026, 7, 20, 9, 30, 0, tzinfo=timezone.utc)
        day1 = datetime(2026, 7, 21, 0, 0, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc)
        cur = _make_mock_cursor(
            fetchone_side_effect=[(last_obs,), (last_mov,), (42,), (7,)],
            fetchall_return=[
                (day1, "scene_intel", 10),
                (day2, "scene_intel", 8),
                (day2, "llm_vision", 2),
            ],
        )
        orig_pool = _install_pool(cur)
        try:
            response = test_client.get("/api/v1/stats/write-health?days=14")
        finally:
            _restore_pool(orig_pool)

        assert response.status_code == 200
        data = response.json()
        assert data["total_observations"] == 42
        assert data["total_movements"] == 7
        assert datetime.fromisoformat(data["last_observation_at"]) == last_obs
        assert datetime.fromisoformat(data["last_movement_at"]) == last_mov
        assert len(data["observations_by_day"]) == 3
        assert data["observations_by_day"][0]["source"] == "scene_intel"
        assert data["observations_by_day"][0]["count"] == 10

    def test_write_health_days_param_respected(self, test_client):
        cur = _make_mock_cursor(
            fetchone_side_effect=[(None,), (None,), (0,), (0,)],
            fetchall_return=[],
        )
        orig_pool = _install_pool(cur)
        try:
            response = test_client.get("/api/v1/stats/write-health?days=30")
        finally:
            _restore_pool(orig_pool)

        assert response.status_code == 200
        last_call = cur.execute.call_args_list[-1]
        assert last_call.args[1] == (30,)

    def test_write_health_days_over_max_rejected(self, test_client):
        cur = _make_mock_cursor(
            fetchone_side_effect=[(None,), (None,), (0,), (0,)],
            fetchall_return=[],
        )
        orig_pool = _install_pool(cur)
        try:
            response = test_client.get("/api/v1/stats/write-health?days=91")
        finally:
            _restore_pool(orig_pool)

        assert response.status_code == 422

    def test_write_health_days_default_is_14(self, test_client):
        cur = _make_mock_cursor(
            fetchone_side_effect=[(None,), (None,), (0,), (0,)],
            fetchall_return=[],
        )
        orig_pool = _install_pool(cur)
        try:
            response = test_client.get("/api/v1/stats/write-health")
        finally:
            _restore_pool(orig_pool)

        assert response.status_code == 200
        last_call = cur.execute.call_args_list[-1]
        assert last_call.args[1] == (14,)
