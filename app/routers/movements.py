from fastapi import APIRouter, HTTPException, status
from app.models.schemas import MovementCreate, MovementResponse, MovementTransitionResponse
from app.services.movement_store import MovementStore
from typing import List

router = APIRouter(prefix="/movements", tags=["movements"])

movement_store = MovementStore()

@router.post("/", response_model=MovementResponse, status_code=status.HTTP_201_CREATED)
async def create_movement(movement: MovementCreate):
    try:
        m_id = await movement_store.create(movement)
        return {**movement.dict(), "id": m_id, "created_at": "2026-04-14T00:00:00Z"} # Simplified
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/transitions", response_model=List[MovementTransitionResponse])
async def get_transitions(
    person_id: str,
    semantic: str = None,
    to_room_id: str = None,
    since_minutes: int = None
):
    # Placeholder for the transition logic
    return []
