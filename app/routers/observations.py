"""API router for scene observations with proper error handling."""

from fastapi import APIRouter, HTTPException, status
from app.models.schemas import ObservationCreate, ObservationResponse, ObservationSearchRequest, ObservationSearchResult
from app.services.observation_store import ObservationStore, ObservationStoreError
from app.services.search import SearchService, SearchServiceError
from app.services.text_embedder import build_text_embedder
from app.config.config import settings
from typing import List

router = APIRouter(prefix="/observations", tags=["observations"])

obs_store = ObservationStore()
search_service = SearchService(
    text_embedder=build_text_embedder(
        enabled=settings.TEXT_EMBEDDING_ENABLED,
        model_name=settings.TEXT_EMBEDDING_MODEL,
        device="cpu",
    )
)


@router.post("/", response_model=ObservationResponse, status_code=status.HTTP_201_CREATED)
async def create_observation(obs: ObservationCreate) -> ObservationResponse:
    """Create a new scene observation.

    Args:
        obs: The observation to create.

    Returns:
        The created observation with ID.

    Raises:
        HTTPException: If the observation cannot be created.
    """
    try:
        obs_id = await obs_store.create(obs)
        return {**obs.model_dump(), "id": obs_id, "created_at": "2026-04-14T00:00:00Z"}
    except ObservationStoreError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/search", response_model=List[ObservationSearchResult])
async def search_observations(search_req: ObservationSearchRequest) -> List[ObservationSearchResult]:
    """Search observations using vector similarity.

    Args:
        search_req: The search request.

    Returns:
        List of matching observations.

    Raises:
        HTTPException: If the search fails.
    """
    try:
        return await search_service.search_observations(search_req)
    except SearchServiceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/prune")
async def prune_older_than(days: int = settings.RETENTION_DAYS) -> dict:
    """Prune observations older than specified days.

    Args:
        days: Number of days to retain.

    Returns:
        Count of deleted observations.

    Raises:
        HTTPException: If the prune operation fails.
    """
    from app.db.connection import db

    pool = db.get_pool()
    query = """
        DELETE FROM scene_observations
        WHERE observed_at < NOW() - INTERVAL '1 day' * $1
        RETURNING id
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, days)
        return {"deleted_count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
