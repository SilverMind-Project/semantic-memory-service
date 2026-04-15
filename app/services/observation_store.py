import asyncpg
from app.db.connection import db
from app.models.schemas import ObservationCreate, ObservationResponse
from typing import List, Optional, Dict, Any

class ObservationStore:
    async def create(self, obs: ObservationCreate) -> int:
        pool = db.get_pool()
        # Note: In a production environment, we'd use a more robust way to handle 
        # the conversion of Python lists/dicts to PostgreSQL JSONB and Array types.
        query = """
            INSERT INTO scene_observations (
                sensor_id, room_id, room_name, observed_at, source, 
                objects_json, persons_count, hazard_flags, description, 
                object_list, workflow_execution_id, media_paths_json, embedding
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id;
        """
        async with pool.acquire() as conn:
            res = await conn.fetchval(
                query,
                obs.sensor_id,
                obs.room_id,
                obs.room_name,
                obs.observed_at,
                obs.source,
                # Converting to JSON string for JSONB insertion via asyncpg
                # In a real scenario, we'd use the driver's native JSON support
                import json; json.dumps(obs.objects_json) if obs.objects_json else None,
                obs.persons_count,
                obs.hazard_flags,
                obs.description,
                obs.object_list,
                obs.workflow_execution_id,
                import json; json.dumps(obs.media_paths_json) if obs.media_paths_json else None,
                obs.embedding
            )
            return res

    async def get_by_id(self, obs_id: int) -> Optional[Dict[str, Any]]:
        pool = db.get_pool()
        query = "SELECT * FROM scene_observations WHERE id = $1"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, obs_id)
            return dict(row) if row else None
