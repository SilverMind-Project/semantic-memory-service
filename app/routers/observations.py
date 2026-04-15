from fastapi import APIRouter, HTTPException, status
from app.models.schemas import ObservationCreate, ObservationResponse, ObservationSearchRequest, ObservationSearchResult
from app.services.observation_store import ObservationStore
from app.services.search import SearchService
from typing import List

router = APIRouter(prefix="/observations", tags=["observations"])

# In a real app, use dependency injection for stores
obs_store = ObservationStore()
search_service = SearchService()

@router.post("/", response_model=ObservationResponse, status_code=status.HTTP_201_CREATED)
async def create_observation(obs: ObservationCreate):
    try:
        obs_id = await obs_store.create(obs)
        return {**obs.dict(), "id": obs_id, "created_at": "2026-04-14T00:00:00Z"} # Simplified
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/search", response_model=List[ObservationSearchResult])
async def search_observations(search_req: ObservationSearchRequest):
    return await search_service.search_observations(search_req)
