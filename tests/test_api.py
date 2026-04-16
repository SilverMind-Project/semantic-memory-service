import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from app.main import app

@pytest.fixture(autouse=True)
def setup_db():
    # Mock database connection to avoid needing real PostgreSQL
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    async_mock = AsyncMock()
    async_mock.__aenter__ = AsyncMock(return_value=mock_conn)
    async_mock.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=async_mock)
    mock_conn.execute = AsyncMock(return_value=None)
    
    with patch("app.db.connection.db.get_pool", return_value=mock_pool):
        with patch("app.db.connection.db.connect", new_callable=AsyncMock):
            with patch("app.db.connection.db.disconnect", new_callable=AsyncMock):
                yield

def test_health_check():
    with TestClient(app=app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "semantic-memory-service"}
