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
        """Upsert object presence record.

        Args:
            room_id: The room ID.
            object_label: The object label.
            observation_id: The observation ID.

        Raises:
            ObjectPresenceStoreError: If the upsert fails.
        """
        pool = db.get_pool()
        query = """
            INSERT INTO object_presence (
                id, room_id, object_label, first_seen_at, last_seen_at,
                observation_count, last_observation_id
            )
            VALUES (
                (SELECT COALESCE(MAX(id), 0) + 1 FROM object_presence),
                $1, $2, NOW(), NOW(), 1, $3
            )
            ON CONFLICT (room_id, object_label)
            DO UPDATE SET
                last_seen_at = EXCLUDED.last_seen_at,
                observation_count = object_presence.observation_count + 1,
                last_observation_id = EXCLUDED.last_observation_id;
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(query, room_id, object_label, observation_id)
        except Exception as e:
            raise ObjectPresenceStoreError(f"Failed to upsert presence: {e}")

    async def get_by_room(
        self,
        room_id: str,
        since_minutes: int,
    ) -> list[dict]:
        """Get object presence for a room within a time window.

        Args:
            room_id: The room ID.
            since_minutes: Time window in minutes.

        Returns:
            List of object presence records.

        Raises:
            ObjectPresenceStoreError: If the query fails.
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
                return [dict(row) for row in rows]
        except Exception as e:
            raise ObjectPresenceStoreError(f"Failed to get object presence: {e}")

    async def delete_old_records(self, days: int) -> int:
        """Delete object presence records older than specified days.

        Args:
            days: Number of days to retain.

        Returns:
            Number of deleted records.

        Raises:
            ObjectPresenceStoreError: If the delete fails.
        """
        pool = db.get_pool()
        query = """
            DELETE FROM object_presence
            WHERE last_seen_at < NOW() - INTERVAL '1 day' * $1
            RETURNING id
        """
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, days)
                return len(rows)
        except Exception as e:
            raise ObjectPresenceStoreError(f"Failed to delete old records: {e}")
