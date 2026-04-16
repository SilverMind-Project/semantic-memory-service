"""Store layer for person movements with proper error handling."""

from app.db.connection import db
from app.models.schemas import MovementCreate, MovementTransitionResponse
from typing import List


class MovementStoreError(Exception):
    """Base exception for movement store errors."""

    pass


class MovementStore:
    """Store layer for person movements."""

    async def create(self, movement: MovementCreate) -> int:
        """Create a new movement record.

        Args:
            movement: The movement to create.

        Returns:
            The ID of the created movement.

        Raises:
            MovementStoreError: If the movement cannot be created.
        """
        pool = db.get_pool()
        query = """
            INSERT INTO person_movements (
                person_id, person_name, sensor_id, from_room_id, to_room_id,
                from_room_name, to_room_name, direction_raw, direction_semantic,
                confidence, observed_at, observation_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id;
        """
        try:
            async with pool.acquire() as conn:
                movement_id = await conn.fetchval(
                    query,
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
                )

            if movement_id is None:
                raise MovementStoreError("Failed to create movement")

            return movement_id
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
            MovementStoreError: If the query fails.
        """
        pool = db.get_pool()
        params: list = [person_id]
        param_idx = 2

        query = """
            SELECT person_id, person_name, direction_semantic, to_room_name,
                   confidence, observed_at
            FROM person_movements
            WHERE person_id = $1
        """

        if semantic:
            query += f" AND direction_semantic = ${param_idx}"
            params.append(semantic)
            param_idx += 1

        if to_room_id:
            query += f" AND to_room_id = ${param_idx}"
            params.append(to_room_id)
            param_idx += 1

        if since_minutes:
            query += f" AND observed_at >= NOW() - INTERVAL '{since_minutes} minutes'"

        query += " ORDER BY observed_at DESC LIMIT 50"

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [
                    MovementTransitionResponse(
                        person_id=r["person_id"],
                        person_name=r["person_name"],
                        direction_semantic=r["direction_semantic"],
                        to_room_name=r["to_room_name"],
                        confidence=r["confidence"],
                        observed_at=r["observed_at"],
                    )
                    for r in rows
                ]
        except Exception as e:
            raise MovementStoreError(f"Failed to get transitions: {e}")
