from app.db.connection import db
from app.models.schemas import MovementCreate, MovementTransitionResponse
from typing import List, Optional

class MovementStore:
    async def create(self, movement: MovementCreate) -> int:
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
        async with pool.acquire() as conn:
            res = await conn.fetchval(
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
                movement.observation_id
            )
            return res

    async def get_transitions(
        self, 
        person_id: str, 
        semantic: Optional[str], 
        to_room_id: Optional[str], 
        since_minutes: Optional[int]
    ) -> List[MovementTransitionResponse]:
        pool = db.get_pool()
        query = """
            SELECT 
                person_id, person_name, direction_semantic, to_room_name, 
                confidence, observed_at
            FROM person_movements
            WHERE person_id = $1
        """
        params = [person_id]
        
        if semantic:
            params.append(semantic)
            query += f" AND direction_semantic = ${len(params)}"
            
        if to_room_id:
            params.append(to_room_id)
            query += f" AND to_room_id = ${len(params)}"
            
        if since_minutes:
            query += f" AND observed_at >= NOW() - INTERVAL '{since_minutes} minutes'"
            
        query += " ORDER BY observed_at DESC LIMIT 50"

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                MovementTransitionResponse(
                    person_id=r['person_id'],
                    person_name=r['person_name'],
                    direction_semantic=r['direction_semantic'],
                    to_room_name=r['to_room_name'],
                    confidence=r['confidence'],
                    observed_at=r['observed_at']
                ) for r
                in rows
            ]
