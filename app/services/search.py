"""Search service for semantic observations with proper error handling."""

from app.config.config import settings
from app.db.connection import db
from app.models.schemas import ObservationSearchRequest, ObservationSearchResult
from app.services.text_embedder import TextEmbedder, build_text_embedder
from typing import List


class SearchServiceError(Exception):
    """Base exception for search service errors."""

    pass


class SearchService:
    """Service for searching observations with vector similarity.

    Supports three search modes:
    1. Image-only: query_embedding for visual similarity
    2. Text-only: query_text for semantic description similarity
    3. Hybrid: Both queries for combined visual + semantic search
    """

    def __init__(self, text_embedder: TextEmbedder | None = None):
        """Initialize search service with optional text embedder.

        Args:
            text_embedder: Optional pre-initialized text embedder.
                          If None, creates one from settings.
        """
        self._text_embedder = text_embedder or build_text_embedder(
            enabled=settings.TEXT_EMBEDDING_ENABLED,
            model_name=settings.TEXT_EMBEDDING_MODEL,
            device="cpu",
        )

    async def search_observations(
        self,
        search_req: ObservationSearchRequest,
    ) -> List[ObservationSearchResult]:
        """Search observations using vector similarity and filters.

        Supports text-only, image-only, or hybrid search:
        - Text-only: Uses description_embedding with HNSW index
        - Image-only: Uses embedding (CLIP) with HNSW index
        - Hybrid: Combines both similarity scores

        Args:
            search_req: The search request with query, filters, and parameters.

        Returns:
            List of matching observations with similarity scores.

        Raises:
            SearchServiceError: If the search fails.
        """
        pool = db.get_pool()

        # Build SELECT clause with similarity columns
        select_clauses = [
            "SELECT id, observed_at, room_name, description, hazard_flags, object_list"
        ]
        similarity_columns = []

        has_image_query = bool(search_req.query_embedding)
        has_text_query = bool(search_req.query_text and self._text_embedder.is_available)

        if has_image_query:
            similarity_columns.append("embedding <=> $1 AS image_similarity")

        if has_text_query:
            similarity_columns.append("description_embedding <=> $1 AS text_similarity")

        if similarity_columns:
            select_clauses.append(", ".join(similarity_columns))

        # Build FROM and WHERE clauses
        query_parts = ["FROM scene_observations WHERE 1=1"]
        params: list = []
        param_idx = 1

        # Apply filters
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

        # Add similarity thresholds
        if has_image_query:
            params.append(search_req.query_embedding)
            params.append(search_req.similarity_threshold)
            query_parts.append(
                f"AND embedding <=> ${param_idx - 1} <= (1 - ${param_idx})"
            )

        if has_text_query:
            # Generate text embedding for query
            query_text_embedding = self._text_embedder.embed(search_req.query_text or "")
            if query_text_embedding:
                params.append(query_text_embedding)
                params.append(search_req.similarity_threshold)
                query_parts.append(
                    f"AND description_embedding <=> ${param_idx - 1} <= (1 - ${param_idx})"
                )

        # Build ORDER BY and LIMIT
        # Prioritize by available similarity scores (hybrid ranking)
        if has_image_query and has_text_query:
            # Hybrid ranking: average of normalized similarities
            query_parts.append(
                "ORDER BY (image_similarity + text_similarity) / 2 DESC"
            )
        elif has_image_query:
            query_parts.append("ORDER BY image_similarity DESC")
        elif has_text_query:
            query_parts.append("ORDER BY text_similarity DESC")
        else:
            # No similarity search, just filters
            query_parts.append("ORDER BY observed_at DESC")

        # Add LIMIT
        params.append(search_req.limit)
        limit_param = param_idx
        if has_image_query or has_text_query:
            # Count parameters added for similarity
            param_count = 2 if (has_image_query and has_text_query) else 2
            limit_param = param_idx + param_count - 1
        query_parts.append(f"LIMIT ${limit_param}")

        query = " ".join(select_clauses + query_parts)

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                results = []
                for r in rows:
                    result_dict = {
                        "id": r["id"],
                        "observed_at": r["observed_at"],
                        "room_name": r["room_name"],
                        "description": r["description"],
                        "hazard_flags": r["hazard_flags"] or [],
                        "object_list": r["object_list"] or [],
                    }
                    # Add similarity scores if present
                    if has_image_query:
                        result_dict["image_similarity"] = (
                            float(r["image_similarity"]) if r["image_similarity"] else None
                        )
                    if has_text_query:
                        result_dict["text_similarity"] = (
                            float(r["text_similarity"]) if r["text_similarity"] else None
                        )
                    results.append(ObservationSearchResult(**result_dict))
                return results
        except Exception as e:
            raise SearchServiceError(f"Search failed: {e}")
