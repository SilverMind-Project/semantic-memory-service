"""Store layer for person movements with proper error handling."""

from datetime import datetime

from app.db.connection import db
from app.models.schemas import MovementCreate, MovementTransitionResponse


class MovementStoreError(Exception):
    """Base exception for movement store errors."""
    pass


class MovementStore:
    """Store layer for person movements."""

    async def create(self, movement: MovementCreate) -> tuple[int, datetime]:
        """Create a new movement record.

        Returns:
            (id, created_at) of the new movement.
        """
        pool = db.get_pool()
        query = """
            INSERT INTO person_movements (
                person_id, person_name, sensor_id, from_room_id, to_room_id,
                from_room_name, to_room_name, direction_raw, direction_semantic,
                confidence, observed_at, observation_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at;
        """
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        query,
                        (
                            movement.person_id,
                            movement.person_name,
                            movement.sensor_id,
                            movement.from_room_id,
                            movement.to_room_id,
                            movement.from_room_name,
                            movement.to_room_name,
                            movement.direction_raw,
                            movement.direction_semantic,
                            movement.confidence,
                            movement.observed_at,
                            movement.observation_id,
                        ),
                    )
                    row = await cur.fetchone()

            if row is None:
                raise MovementStoreError("Failed to create movement")

            return row[0], row[1]
        except MovementStoreError:
            raise
        except Exception as e:
            raise MovementStoreError(f"Failed to create movement: {e}")

    async def get_transitions(
        self,
        person_id: str,
        semantic: str | None = None,
        to_room_id: str | None = None,
        since_minutes: int | None = None,
    ) -> list[MovementTransitionResponse]:
        """Get movement transitions for a person."""
        pool = db.get_pool()
        params: list = [person_id]

        query = """
            SELECT id, person_id, person_name, from_room_id, to_room_id,
                   from_room_name, to_room_name, direction_semantic,
                   confidence, observed_at, observation_id
            FROM person_movements
            WHERE person_id = %s
        """

        if semantic:
            query += " AND direction_semantic = %s"
            params.append(semantic)

        if to_room_id:
            query += " AND to_room_id = %s"
            params.append(to_room_id)

        if since_minutes:
            query += " AND observed_at >= NOW() - INTERVAL '1 minute' * %s"
            params.append(since_minutes)

        query += " ORDER BY observed_at DESC LIMIT 50"

        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    rows = await cur.fetchall()
                    cols = [desc[0] for desc in (cur.description or [])]
                    return [
                        MovementTransitionResponse(**dict(zip(cols, row)))
                        for row in rows
                    ]
        except Exception as e:
            raise MovementStoreError(f"Failed to get transitions: {e}")
