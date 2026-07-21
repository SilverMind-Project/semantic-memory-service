"""Store layer for write-health aggregate stats, with proper error handling."""

from app.db.connection import db
from app.models.schemas import ObservationsByDay, WriteHealthResponse


class StatsStoreError(Exception):
    """Base exception for stats store errors."""
    pass


class StatsStore:
    """Aggregate read layer over scene_observations and person_movements.

    Read-only; never mutates data. All queries run in one connection so the
    counts and the day-bucketed breakdown reflect the same snapshot.
    """

    async def write_health(self, days: int) -> WriteHealthResponse:
        """Return write recency and volume across observations and movements."""
        pool = db.get_pool()
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT max(observed_at) FROM scene_observations")
                    last_observation_row = await cur.fetchone()

                    await cur.execute("SELECT max(observed_at) FROM person_movements")
                    last_movement_row = await cur.fetchone()

                    await cur.execute("SELECT count(*) FROM scene_observations")
                    total_observations_row = await cur.fetchone()

                    await cur.execute("SELECT count(*) FROM person_movements")
                    total_movements_row = await cur.fetchone()

                    await cur.execute(
                        """
                        SELECT date_trunc('day', observed_at) AS day, source, count(*) AS count
                        FROM scene_observations
                        WHERE observed_at >= NOW() - INTERVAL '1 day' * %s
                        GROUP BY 1, 2
                        ORDER BY 1 DESC
                        """,
                        (days,),
                    )
                    by_day_rows = await cur.fetchall()

            return WriteHealthResponse(
                last_observation_at=last_observation_row[0] if last_observation_row else None,
                last_movement_at=last_movement_row[0] if last_movement_row else None,
                total_observations=(
                    total_observations_row[0] if total_observations_row else 0
                ),
                total_movements=total_movements_row[0] if total_movements_row else 0,
                observations_by_day=[
                    ObservationsByDay(day=row[0], source=row[1], count=row[2])
                    for row in by_day_rows
                ],
            )
        except Exception as e:
            raise StatsStoreError(f"Failed to compute write health: {e}")
