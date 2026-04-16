"""Search service for semantic observations with proper error handling."""

from app.db.connection import db
from app.models.schemas import ObservationSearchRequest, ObservationSearchResult
from typing import List


class SearchServiceError(Exception):
    """Base exception for search service errors."""

    pass


class SearchService:
    """Service for searching observations with vector similarity."""

    async def search_observations(
        self,
        search_req: ObservationSearchRequest,
    ) -> List[ObservationSearchResult]:
        """Search observations using vector similarity and filters.

        Args:
            search_req: The search request with query, filters, and parameters.

        Returns:
            List of matching observations with similarity scores.

        Raises:
            SearchServiceError: If the search fails.
        """
        pool = db.get_pool()

        query_parts = [
            "SELECT id, observed_at, room_name, description, hazard_flags, object_list"
        ]
        select_embedding = False
        if search_req.query_embedding:
            query_parts.append(", embedding <=> $1 AS similarity")
            select_embedding = True

        query_parts.append("FROM scene_observations WHERE 1=1")
        params: list = []
        param_idx = 1

        if search_req.room_id:
            params.append(search_req.room_id)
            query_parts.append(f"AND room_id = ${param_idx}")
            param_idx += 1

        if search_req.since_minutes:
            query_parts.append(
                f"AND observed_at >= NOW() - INTERVAL '{search_req.since_minutes} minutes'"
            )

        if search_req.objects_any:
            params.append(search_req.objects_any)
            query_parts.append(f"AND object_list && ${param_idx}")
            param_idx += 1

        if search_req.hazard_flags_any:
            params.append(search_req.hazard_flags_any)
            query_parts.append(f"AND hazard_flags && ${param_idx}")
            param_idx += 1

        if search_req.query_embedding:
            params.append(search_req.query_embedding)
            params.append(search_req.similarity_threshold)
            query_parts.append(
                f"AND embedding <=> ${param_idx - 1} <= (1 - ${param_idx})"
            )

        query_parts.append("ORDER BY observed_at DESC")
        if select_embedding:
            query_parts.append(f"LIMIT ${param_idx + 1}")
        else:
            query_parts.append(f"LIMIT ${param_idx}")
        params.append(search_req.limit)

        query = " ".join(query_parts)

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                results = [
                    ObservationSearchResult(
                        id=r["id"],
                        observed_at=r["observed_at"],
                        room_name=r["room_name"],
                        description=r["description"],
                        hazard_flags=r["hazard_flags"] or [],
                        object_list=r["object_list"] or [],
                        similarity=float(r["similarity"]) if select_embedding else None,
                    )
                    for r in rows
                ]
                return results
        except Exception as e:
            raise SearchServiceError(f"Search failed: {e}")
