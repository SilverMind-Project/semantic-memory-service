"""Store layer for scene observations with proper error handling."""

from app.db.connection import db
from app.models.schemas import ObservationCreate
import json
from typing import Optional


class ObservationStoreError(Exception):
    """Base exception for observation store errors."""

    pass


class ObservationStore:
    """Store layer for scene observations."""

    async def create(self, obs: ObservationCreate) -> int:
        """Create a new scene observation.

        Args:
            obs: The observation to create.

        Returns:
            The ID of the created observation.

        Raises:
            ObservationStoreError: If the observation cannot be created.
        """
        pool = db.get_pool()
        query = """
            INSERT INTO scene_observations (
                sensor_id, room_id, room_name, observed_at, source,
                objects_json, persons_count, hazard_flags, description,
                description_embedding, object_list, workflow_execution_id,
                media_paths_json, embedding
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING id;
        """
        try:
            objects_json = json.dumps(obs.objects_json) if obs.objects_json else None
            media_paths_json = json.dumps(obs.media_paths_json) if obs.media_paths_json else None

            async with pool.acquire() as conn:
                obs_id = await conn.fetchval(
                    query,
                    obs.sensor_id,
                    obs.room_id,
                    obs.room_name,
                    obs.observed_at,
                    obs.source,
                    objects_json,
                    obs.persons_count,
                    obs.hazard_flags,
                    obs.description,
                    obs.description_embedding,
                    obs.object_list,
                    obs.workflow_execution_id,
                    media_paths_json,
                    obs.embedding,
                )

            if obs_id is None:
                raise ObservationStoreError("Failed to create observation")

            return obs_id
        except ObservationStoreError:
            raise
        except Exception as e:
            raise ObservationStoreError(f"Failed to create observation: {e}")

    async def get_by_id(self, obs_id: int) -> Optional[dict]:
        """Retrieve an observation by ID.

        Args:
            obs_id: The observation ID.

        Returns:
            The observation as a dictionary, or None if not found.

        Raises:
            ObservationStoreError: If the database query fails.
        """
        pool = db.get_pool()
        query = "SELECT * FROM scene_observations WHERE id = $1"
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, obs_id)
                return dict(row) if row else None
        except Exception as e:
            raise ObservationStoreError(f"Failed to retrieve observation: {e}")
