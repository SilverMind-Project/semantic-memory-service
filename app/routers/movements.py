"""API router for person movements with proper error handling."""

from fastapi import APIRouter, HTTPException, status
from app.models.schemas import MovementCreate, MovementResponse, MovementTransitionResponse
from app.services.movement_store import MovementStore, MovementStoreError
from typing import List

router = APIRouter(prefix="/movements", tags=["movements"])

movement_store = MovementStore()


@router.post("/", response_model=MovementResponse, status_code=status.HTTP_201_CREATED)
async def create_movement(movement: MovementCreate) -> MovementResponse:
    """Create a new movement record.

    Args:
        movement: The movement to create.

    Returns:
        The created movement with ID.

    Raises:
        HTTPException: If the movement cannot be created.
    """
    try:
        m_id = await movement_store.create(movement)
        return {**movement.model_dump(), "id": m_id, "created_at": "2026-04-14T00:00:00Z"}
    except MovementStoreError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/transitions", response_model=List[MovementTransitionResponse])
async def get_transitions(
    person_id: str,
    semantic: str | None = None,
    to_room_id: str | None = None,
    since_minutes: int | None = None,
) -> List[MovementTransitionResponse]:
    """Get movement transitions for a person.

    Args:
        person_id: The person's ID.
        semantic: Optional semantic direction filter.
        to_room_id: Optional target room ID filter.
        since_minutes: Optional time window in minutes.

    Returns:
        List of movement transitions.

    Raises:
        HTTPException: If the query fails.
    """
    try:
        return await movement_store.get_transitions(person_id, semantic, to_room_id, since_minutes)
    except MovementStoreError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
