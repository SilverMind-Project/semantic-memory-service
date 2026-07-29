"""API router for object presence."""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status

from app.services.object_presence import ObjectPresenceStore, ObjectPresenceStoreError

router = APIRouter(prefix="/objects", tags=["objects"])

presence_store = ObjectPresenceStore()


@router.get("/{room_id}/recent")
async def get_recent_objects(room_id: str, since_minutes: int = 60) -> list[dict]:
    """Get recent object presence in a room."""
    try:
        rows = await presence_store.get_by_room(room_id, since_minutes)
    except ObjectPresenceStoreError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    if not rows:
        return []

    # last_seen_at is returned alongside the relative form: consumers model an
    # absolute timestamp, and deriving one back from a rounded "minutes ago"
    # loses the original instant.
    now = datetime.now(UTC)
    return [
        {
            "label": r["object_label"],
            "last_seen_at": r["last_seen_at"],
            "last_seen_minutes_ago": round(
                (now - _as_utc(r["last_seen_at"])).total_seconds() / 60
            ),
            "observation_count": r["observation_count"],
        }
        for r in rows
    ]


def _as_utc(value: datetime) -> datetime:
    """Treat a naive timestamp as UTC; the column is TIMESTAMPTZ."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)
