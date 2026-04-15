from app.db.connection import db
from app.models.schemas import ObservationCreate, ObservationSearchRequest, ObservationSearchResult
from typing import List, Dict, Any
import json


class ObservationStore:
    async def create_observation(self, obs: ObservationCreate) -> int:
        query = """
            INSERT INTO observations (
                room_name, observed_at, source, objects_json,
                persons_count, hazard_flags, description,
                object_list, workflow_execution_id, media_paths_json, embedding
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
            ) RETURNING id
        """
        obs_id = await db.fetchval(
            query,
            obs.room_name,
            obs.observed_at,
            obs.source,
            json.dumps(obs.objects_json) if obs.objects_json else None,
            obs.persons_count,
            obs.hazard_flags,
            obs.description,
            json.dumps(obs.object_list) if obs.object_list else None,
            obs.workflow_execution_id,
            json.dumps(obs.media_paths_json) if obs.media_paths_json else None,
            json.dumps(obs.embedding) if obs.embedding else None,
        )
        return obs_id

    async def search_observations(self, query: ObservationSearchRequest) -> List[ObservationSearchResult]:
        args: List[Any] = []
        conditions = []

        if query.room_name:
            conditions.append("room_name ILIKE %s")
            args.append(f"%{query.room_name}%")

        if query.start_date:
            conditions.append("observed_at >= %s")
            args.append(query.start_date)

        if query.end_date:
            conditions.append("observed_at <= %s")
            args.append(query.end_date)

        if query.objects:
            conditions.append("objects_json @> %s")
            args.append(json.dumps(query.objects))

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT id, room_name, observed_at, source, objects_json,
                   persons_count, hazard_flags, description,
                   object_list, workflow_execution_id, media_paths_json, embedding
            FROM observations
            WHERE {where_clause}
            ORDER BY observed_at DESC
            LIMIT 100
        """
        return await db.fetch(sql, *args)


class MovementStore:
    async def create_movement(self, move: Dict[str, Any], observation_id: int):
        query = """
            INSERT INTO movements (
                person_id, from_room, to_room, timestamp, observation_id
            ) VALUES (%s, %s, %s, %s, %s)
        """
        await db.execute(query, move["person_id"], move["from_room"], move["to_room"], move["timestamp"], observation_id)

    async def get_movements(self, room: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        query = """
            SELECT m.*, o.room_name as source_room
            FROM movements m
            JOIN observations o ON m.observation_id = o.id
            WHERE m.from_room = %s OR m.to_room = %s
            AND m.timestamp >= %s AND m.timestamp <= %s
            ORDER BY m.timestamp DESC
        """
        return await db.fetch(query, room, room, start_time, end_time)


class ObjectPresenceStore:
    async def record_object_presence(self, room: str, object_name: str, timestamp: str, observation_id: int):
        query = """
            INSERT INTO object_presence (room, object_name, timestamp, observation_id)
            VALUES (%s, %s, %s, %s)
        """
        await db.execute(query, room, object_name, timestamp, observation_id)

    async def get_objects_in_room(self, room: str, since: str) -> List[Dict[str, Any]]:
        query = """
            SELECT DISTINCT object_name, MAX(timestamp) as last_seen
            FROM object_presence
            WHERE room = %s AND timestamp >= %s
            GROUP BY object_name
            ORDER BY last_seen DESC
        """
        return await db.fetch(query, room, since)
