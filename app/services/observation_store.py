"""Store layer for scene observations with proper error handling."""

import json
from datetime import datetime

from app.db.connection import db
from app.models.schemas import ObservationCreate


class ObservationStoreError(Exception):
    """Base exception for observation store errors."""
    pass


class ObservationStore:
    """Store layer for scene observations."""

    async def create(self, obs: ObservationCreate) -> tuple[int, datetime]:
        """Create a new scene observation.

        Returns:
            (id, created_at) of the new observation.
        """
        pool = db.get_pool()
        query = """
            INSERT INTO scene_observations (
                sensor_id, room_id, room_name, observed_at, source,
                objects_json, persons_count, hazard_flags, description,
                description_embedding, object_list, workflow_execution_id,
                media_paths_json, embedding, person_id, kind
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at;
        """
        try:
            objects_json = json.dumps(obs.objects_json) if obs.objects_json else None
            media_paths_json = json.dumps(obs.media_paths_json) if obs.media_paths_json else None

            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        query,
                        (
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
                            obs.person_id,
                            obs.kind,
                        ),
                    )
                    row = await cur.fetchone()

            if row is None:
                raise ObservationStoreError("Failed to create observation")

            return row[0], row[1]
        except ObservationStoreError:
            raise
        except Exception as e:
            raise ObservationStoreError(f"Failed to create observation: {e}")

    async def get_by_id(self, obs_id: int) -> dict | None:
        """Retrieve an observation by ID."""
        pool = db.get_pool()
        query = "SELECT * FROM scene_observations WHERE id = %s"
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (obs_id,))
                    row = await cur.fetchone()
                    if row is None:
                        return None
                    cols = [desc[0] for desc in (cur.description or [])]
                    return dict(zip(cols, row))
        except Exception as e:
            raise ObservationStoreError(f"Failed to retrieve observation: {e}")
