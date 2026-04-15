import pytest
from httpx import AsyncClient
from app.main import app
from app.db.connection import db

@pytest.fixture(autouse=True)
async def setup_db():
    # In a real test environment, we would use a separate test database
    # and run migrations. For this unit test, we ensure the pool is available.
    await db.connect()
    yield
    await db.disconnect()

@pytest

async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "semantic-memory-service"}

@pytest.mark.asyncio
async def test_create_observation_invalid_data():
    # Test with missing required fields
    payload = {"sensor_id": "test-sensor"} 
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/v1/observations/", json=payload)
    
    assert response.status_code == 422 # Unprocessable Entity (Pydantic validation error)
