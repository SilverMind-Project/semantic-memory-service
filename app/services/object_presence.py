"""Store layer for object presence tracking with proper error handling."""

from app.db.connection import db


class ObjectPresenceStoreError(Exception):
    """Base exception for object presence store errors."""
    pass


class ObjectPresenceStore:
    """Store layer for object presence tracking."""

    async def upsert_presence(
        self,
        room_id: str,
        object_label: str,
        observation_id: int,
    ) -> None:
        """Upsert object presence record."""
        pool = db.get_pool()
        query = """
            INSERT INTO object_presence (
                room_id, object_label, first_seen_at, last_seen_at,
                observation_count, last_observation_id
            )
            VALUES (
                %s, %s, NOW(), NOW(), 1, %s
            )
            ON CONFLICT (room_id, object_label)
            DO UPDATE SET
                last_seen_at = EXCLUDED.last_seen_at,
                observation_count = object_presence.observation_count + 1,
                last_observation_id = EXCLUDED.last_observation_id;
        """
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (room_id, object_label, observation_id))
        except Exception as e:
            raise ObjectPresenceStoreError(f"Failed to upsert presence: {e}")

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
