"""Store layer for object presence tracking with proper error handling."""

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from app.db.connection import db


class ObjectPresenceStoreError(Exception):
    """Base exception for object presence store errors."""
    pass


class ObjectPresenceStore:
    """Store layer for object presence tracking."""

    # Timestamps come from the caller, not NOW(): an observation carries the
    # time the scene was captured, which is not the time this row is written.
    UPSERT_SQL = """
        INSERT INTO object_presence (
            room_id, object_label, first_seen_at, last_seen_at,
            observation_count, last_observation_id
        )
        VALUES (
            %s, %s, %s, %s, 1, %s
        )
        ON CONFLICT (room_id, object_label)
        DO UPDATE SET
            first_seen_at = LEAST(object_presence.first_seen_at, EXCLUDED.first_seen_at),
            last_seen_at = GREATEST(object_presence.last_seen_at, EXCLUDED.last_seen_at),
            observation_count = object_presence.observation_count + 1,
            last_observation_id = EXCLUDED.last_observation_id;
    """

    async def upsert_presence(
        self,
        room_id: str,
        object_label: str,
        observation_id: int,
        observed_at: datetime | None = None,
    ) -> None:
        """Upsert object presence record on its own connection."""
        pool = db.get_pool()
        seen_at = observed_at or datetime.now(UTC)
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        self.UPSERT_SQL,
                        (room_id, object_label, seen_at, seen_at, observation_id),
                    )
        except Exception as e:
            raise ObjectPresenceStoreError(f"Failed to upsert presence: {e}")

    @classmethod
    async def upsert_many_on_cursor(
        cls,
        cur: Any,
        *,
        room_id: str,
        object_labels: Iterable[str],
        observation_id: int,
        observed_at: datetime,
    ) -> None:
        """Upsert presence for several labels reusing an open cursor.

        Sharing the observation's cursor keeps the two writes in one
        transaction: an observation whose objects never registered is a silent
        half-write, and the caller has no transaction to roll back into.
        """
        seen = {label for label in object_labels if label}
        for label in sorted(seen):
            await cur.execute(
                cls.UPSERT_SQL,
                (room_id, label, observed_at, observed_at, observation_id),
            )

    async def get_by_room(self, room_id: str, since_minutes: int) -> list[dict]:
        """Get object presence for a room within a time window."""
        pool = db.get_pool()
        query = """
            SELECT object_label, last_seen_at, observation_count
            FROM object_presence
            WHERE room_id = %s
              AND last_seen_at >= NOW() - INTERVAL '1 minute' * %s
            ORDER BY last_seen_at DESC
        """
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (room_id, since_minutes))
                    rows = await cur.fetchall()
                    cols = [desc[0] for desc in (cur.description or [])]
                    return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            raise ObjectPresenceStoreError(f"Failed to get object presence: {e}")

    async def delete_old_records(self, days: int) -> int:
        """Delete object presence records older than specified days."""
        pool = db.get_pool()
        query = """
            DELETE FROM object_presence
            WHERE last_seen_at < NOW() - INTERVAL '1 day' * %s
            RETURNING id
        """
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (days,))
                    rows = await cur.fetchall()
                    return len(rows)
        except Exception as e:
            raise ObjectPresenceStoreError(f"Failed to delete old records: {e}")
