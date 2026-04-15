from app.db.connection import db

class ObjectPresenceStore:
    async def upsert_presence(
        self, 
        room_id: str, 
        object_label: str, 
        observation_id: int
    ) -> None:
        pool = db.get_pool()
        # Fixed the typo 'object_lat' to 'object_label'
        # Note: This assumes a sequence or auto-incrementing primary key exists for object_presence
        # or that we handle ID generation. In the schema, it was defined as BIGSERIAL/BIGINT.
        # For simplicity in this skeleton, we'll use a standard upsert.
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
        async with pool.acquire() as conn:
            await conn.execute(query, room_id, object_label, observation_id)
