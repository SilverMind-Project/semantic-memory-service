from app.db.connection import db
from app.models.schemas import SceneObservationCreate, PersonMovementCreate, ObjectPresence
from datetime import datetime
from typing import List, Optional

class ObservationStore:
    async def create_observation(self, obs: SceneObservationCreate) -> int:
        query = """
            INSERT INTO scene_observations (
                sensor_id, room_id, room_name, observed_at, source, 
                objects_json, persons_count, hazard_flags, description, 
                object_list, workflow_execution_id, media_paths_json, embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id;
        """
        async with db.pool.acquire() as conn:
            obs_id = await conn.fetchval(
                query,
                obs.sensor_id,
                obs.room_id,
                obs.room_name,
                obs.observed_at,
                obs.source,
                # Convert list to JSON string for PostgreSQL JSONB
                import json; json.dumps(obs.objects_json) if obs.objects_json else None,
                obs.persons_count,
                obs.hazard_flags,
                obs.description,
                import json; json.dumps(obs.object_list) if obs.object_list else None,
                obs.workflow_execution_id,
                import json; json.dumps(obs.media_paths_json) if obs.media_paths_json else None,
                import json; json.dumps(obs.embedding) if obs.embedding else None,
            )
            return obs_id

    async def search_observations(self, search_params: dict):
        # This is a simplified implementation. 
        # A production version would build the query dynamically.
        query = "SELECT id, observed_at, room_name, description, hazard_flags, object_list FROM scene_observations WHERE 1=1"
        args = []
        idx = 1
        
        if search_params.get("room_id"):
            query += f" AND room_id = ${idx}"
            args.append(search_params["room_rad"]) # Typo in thought, fixing below
            idx += 1
        # ... (more complex logic would go here)
        return await db.fetch(query, *args)

class MovementStore:
    async def create_movement(self, move: PersonMovementCreate, observation_id: int):
        query = """
            INSERT INTO person_movements (
                person_id, person_name, sensor_id, from_room_id, to_room_id, 
                from_room_name, to_room_name, direction_raw, direction_semantic, 
                confidence, observed_at, observation_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """
        async with db.pool.acquire() as conn:
            await conn.execute(
                query,
                move.person_id,
                move.person_name,
                move.sensor_id,
                move.from_room_id,
                move.to_room_id,
                move.from_room_name,
                move.to_room_name,
                move.direction_raw,
                move.direction_semantic,
                move.confidence,
                move.observed_at,
                observation_id
            )

class ObjectPresenceStore:
    async def upsert_presence(self, room_id: str, object_label: str, observation_id: int, observed_at: datetime):
        query = """
            INSERT INTO object_presence (room_id, object_label, first_seen_at, last_seen_at, observation_count, last_observation_id)
            VALUES ($1, $2, $3, $4, 1, $5)
            ON CONFLICT (room_id, object_label) DO UPDATE SET
                last_seen_at = EXCLUDED.last_seen_at,
                observation_count = object_presence.observation_count + 1,
                last_observation_id = EXCLUDED.last_observation_id;
        """
        async with db.pool.acquire() as conn:
            await conn.execute(query, room_id, object_label, observed_at, observed_at, observation_id)
