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
    """Service for searching observations with vector similarity."""

    def __init__(self, text_embedder: TextEmbedder | None = None):
        self._text_embedder = text_embedder or build_text_embedder(
            enabled=settings.TEXT_EMBEDDING_ENABLED,
            model_name=settings.TEXT_EMBEDDING_MODEL,
            device="cpu",
        )

    async def search_observations(
        self,
        search_req: ObservationSearchRequest,
    ) -> List[ObservationSearchResult]:
        """Search observations using vector similarity and filters."""
        pool = db.get_pool()

        has_image_query = bool(search_req.query_embedding)
        has_text_query = bool(search_req.query_text and self._text_embedder.is_available)

        # Resolve text embedding early so we know if it's actually available
        query_text_embedding = None
        if has_text_query:
            query_text_embedding = self._text_embedder.embed(search_req.query_text or "")
            has_text_query = bool(query_text_embedding)

        # Build SELECT
        similarity_columns = []
        if has_image_query:
            similarity_columns.append("embedding <=> %s AS image_similarity")
        if has_text_query:
            similarity_columns.append("description_embedding <=> %s AS text_similarity")

        select_part = "SELECT id, observed_at, room_name, description, hazard_flags, object_list"
        if similarity_columns:
            select_part += ", " + ", ".join(similarity_columns)

        # Build WHERE + params
        where_clauses = ["1=1"]
        params: list = []

        # Similarity columns come first in params (they appear in SELECT)
        if has_image_query:
            params.append(search_req.query_embedding)
        if has_text_query:
            params.append(query_text_embedding)

        if search_req.room_id:
            where_clauses.append("room_id = %s")
            params.append(search_req.room_id)

        if search_req.since_minutes:
            where_clauses.append(
                f"observed_at >= NOW() - INTERVAL '{search_req.since_minutes} minutes'"
            )

        if search_req.objects_any:
            where_clauses.append("object_list && %s")
            params.append(search_req.objects_any)

        if search_req.hazard_flags_any:
            where_clauses.append("hazard_flags && %s")
            params.append(search_req.hazard_flags_any)

        if has_image_query:
            where_clauses.append("embedding <=> %s <= (1 - %s)")
            params.append(search_req.query_embedding)
            params.append(search_req.similarity_threshold)

        if has_text_query:
            where_clauses.append("description_embedding <=> %s <= (1 - %s)")
            params.append(query_text_embedding)
            params.append(search_req.similarity_threshold)

        # ORDER BY
        if has_image_query and has_text_query:
            order_part = "ORDER BY (image_similarity + text_similarity) / 2 DESC"
        elif has_image_query:
            order_part = "ORDER BY image_similarity DESC"
        elif has_text_query:
            order_part = "ORDER BY text_similarity DESC"
        else:
            order_part = "ORDER BY observed_at DESC"

        params.append(search_req.limit)
        query = (
            f"{select_part} FROM scene_observations "
            f"WHERE {' AND '.join(where_clauses)} "
            f"{order_part} LIMIT %s"
        )

        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    rows = await cur.fetchall()
                    cols = [desc[0] for desc in cur.description]
                    results = []
                    for row in rows:
                        r = dict(zip(cols, row))
                        result_dict = {
                            "id": r["id"],
                            "observed_at": r["observed_at"],
                            "room_name": r["room_name"],
                            "description": r["description"],
                            "hazard_flags": r["hazard_flags"] or [],
                            "object_list": r["object_list"] or [],
                        }
                        if has_image_query:
                            result_dict["image_similarity"] = (
                                float(r["image_similarity"]) if r.get("image_similarity") else None
                            )
                        if has_text_query:
                            result_dict["text_similarity"] = (
                                float(r["text_similarity"]) if r.get("text_similarity") else None
                            )
                        results.append(ObservationSearchResult(**result_dict))
                    return results
        except Exception as e:
            raise SearchServiceError(f"Search failed: {e}")
