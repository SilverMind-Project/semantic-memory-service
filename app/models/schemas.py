from datetime import datetime

from pydantic import BaseModel, Field


# --- Core Entities ---

class ObjectDetection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]
    category: str | None = None
    area_fraction: float | None = None


class ObservationCreate(BaseModel):
    sensor_id: str | None = None
    room_id: str | None = None
    room_name: str | None = None
    observed_at: datetime
    source: str  # 'scene_intel' | 'llm_vision' | 'manual'
    objects_json: list[dict[str, object]] | None = None
    persons_count: int | None = 0
    hazard_flags: list[str] | None = Field(default_factory=list)
    description: str | None = None
    description_embedding: list[float] | None = Field(
        None, description="768-dim text embedding for semantic search"
    )
    object_list: list[str] | None = Field(default_factory=list)
    workflow_execution_id: int | None = None
    media_paths_json: list[str] | None = None
    embedding: list[float] | None = Field(None, description="768-dim CLIP embedding")


class ObservationResponse(ObservationCreate):
    id: int
    created_at: datetime


class ObservationSearchRequest(BaseModel):
    query_embedding: list[float] | None = None
    query_text: str | None = None
    room_id: str | None = None
    since_minutes: int | None = None
    objects_any: list[str] | None = None
    hazard_flags_any: list[str] | None = None
    limit: int = 20
    similarity_threshold: float = 0.75


class ObservationSearchResult(BaseModel):
    id: int
    observed_at: datetime
    room_id: str | None = None
    room_name: str | None = None
    description: str | None = None
    hazard_flags: list[str] = Field(default_factory=list)
    object_list: list[str] = Field(default_factory=list)
    text_similarity: float | None = None
    image_similarity: float | None = None


# --- Movement Entities ---

class MovementCreate(BaseModel):
    person_id: str
    person_name: str | None = None
    sensor_id: str | None = None
    from_room_id: str | None = None
    to_room_id: str | None = None
    from_room_name: str | None = None
    to_room_name: str | None = None
    direction_raw: str | None = None
    direction_semantic: str
    confidence: float
    observed_at: datetime
    observation_id: int | None = None


class MovementResponse(MovementCreate):
    id: int
    created_at: datetime


class MovementTransitionRequest(BaseModel):
    person_id: str
    semantic: str | None = None  # "entering" | "exiting" | "any"
    to_room_id: str | None = None
    since_minutes: int | None = None


class MovementTransitionResponse(BaseModel):
    id: int
    person_id: str
    person_name: str | None = None
    from_room_id: str | None = None
    to_room_id: str | None = None
    from_room_name: str | None = None
    to_room_name: str | None = None
    direction_semantic: str
    confidence: float
    observed_at: datetime
    observation_id: int | None = None


# --- Object Presence ---

class ObjectPresenceResponse(BaseModel):
    room_id: str
    object_label: str
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int
    last_observation_id: int

