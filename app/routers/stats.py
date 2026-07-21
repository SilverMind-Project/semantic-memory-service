"""API router for write-health stats (used by Cognitive Companion's admin surface)."""

from fastapi import APIRouter, HTTPException, Query, status

from app.models.schemas import WriteHealthResponse
from app.services.stats_store import StatsStore, StatsStoreError

router = APIRouter(prefix="/stats", tags=["stats"])

stats_store = StatsStore()


@router.get("/write-health", response_model=WriteHealthResponse)
async def get_write_health(
    days: int = Query(default=14, ge=1, le=90),
) -> WriteHealthResponse:
    """Return write recency and daily volume for observations and movements.

    Args:
        days: Lookback window for the daily observation breakdown (1-90, default 14).

    Raises:
        HTTPException: If the aggregate query fails.
    """
    try:
        return await stats_store.write_health(days)
    except StatsStoreError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
