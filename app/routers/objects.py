"""API router for object presence with proper error handling."""

from fastapi import APIRouter, HTTPException, status
from app.db.connection import db
from datetime import datetime
from typing import List

router = APIRouter(prefix="/objects", tags=["objects"])


@router.get("/{room_id}/recent")
async def get_recent_objects(
    room_id: str,
    since_minutes: int = 60,
) -> List[dict]:
    """Get recent object presence in a room.

    Args:
        room_id: The room ID.
        since_minutes: Time window in minutes (default: 60).

    Returns:
        List of recent object presence records.

    Raises:
        HTTPException: If the query fails.
    """
    pool = db.get_pool()
    query = """
        SELECT object_label, last_seen_at, observation_count
        FROM object_presence
        WHERE room_id = $1
          AND last_seen_at >= NOW() - INTERVAL $2
        ORDER BY last_seen_at DESC
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, room_id, f"{since_minutes} minutes")
            if not rows:
                return []
            now = datetime.now()
            return [
                {
                    "label": r["object_label"],
                    "last_seen_minutes_ago": round(
                        (now - r["last_seen_at"].replace(tzinfo=None)).total_seconds()
                        / 60
                    ),
                    "observation_count": r["observation_count"],
                }
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
