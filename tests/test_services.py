import pytest
from unittest.mock import MagicMock, patch
from app.services.search import SearchService
from app.models.schemas import ObservationSearchRequest

@pytest.mark.asyncio
async def test_search_observations_logic():
    # Mock the database pool and connection
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    # Mock fetch to return a dummy row
    mock_conn.fetch.return_value = [{
        'id': 1,
        'observed_at': '2026-04-14T00:00:00Z',
        'room_name': 'living_room',
        'description': 'A person is sitting on the sofa',
        'hazard_flags': ['none'],
        'object_list': ['person', 'sofa']
    }]

    # Patch the db.get_pool to return our mock pool
    with patch("app.db.connection.db.get_pool", return_value=mock_pool):
        service = SearchService()
        search_req = ObservationSearchRequest(
            room_id="room_1",
            limit=10
        )
        
        results = await service.search_observations(search_req)
        
        assert len(results) == 1
        assert results[0].room_name == 'living_room'
        assert 'person' in results[0].object_list
        
        # Verify that the query was called
        assert mock_conn.fetch.called
