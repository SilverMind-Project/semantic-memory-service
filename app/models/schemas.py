from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Core Entities ---

class ObjectDetection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    category: Optional[str] = None
    area_fraction: Optional[float] = None

class ObservationCreate(BaseModel):
    sensor_id: str
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    observed_at: datetime
    source: str  # 'scene_intel' | 'llm_vision' | 'manual'
    objects_json: Optional[List[Dict[str, Any]]] = None
    persons_count: Optional[int] = 0
    hazard_flags: Optional[List[str]] = Field(default_factory=list)
    description: Optional[str] = None
    description_embedding: Optional[List[float]] = Field(
        None, description="384-dim text embedding for semantic search"
    )
    object_list: Optional[List[str]] = Field(default_factory=list)
    workflow_execution_id: Optional[int] = None
    media_paths_json: Optional[List[str]] = None
    embedding: Optional[List[float]] = Field(None, description="768-dim CLIP embedding")

class ObservationResponse(ObservationCreate):
    id: int
    created_at: datetime

class ObservationSearchRequest(BaseModel):
    query_embedding: Optional[List[float]] = None
    query_text: Optional[str] = None
    room_id: Optional[str] = None
    since_minutes: Optional[int] = None
    objects_any: Optional[List[str]] = None
    hazard_flags_any: Optional[List[str]] = None
    limit: int = 20
    similarity_threshold: float = 0.75

class ObservationSearchResult(BaseModel):
    id: int
    observed_at: datetime
    room_name: Optional[str] = None
    description: Optional[str] = None
    hazard_flags: List[str] = Field(default_factory=list)
    object_list: List[str] = Field(default_factory=list)
    text_similarity: Optional[float] = None
    image_similarity: Optional[float] = None

# --- Movement Entities ---

class MovementCreate(BaseModel):
    person_id: str
    person_name: Optional[str] = None
    sensor_id: str
    from_room_id: Optional[str] = None
    to_room_id: Optional[str] = None
    from_room_name: Optional[str] = None
    to_room_name: Optional[str] = None
    direction_raw: str
    direction_semantic: str
    confidence: float
    observed_at: datetime
    observation_id: Optional[int] = None

class MovementResponse(MovementCreate):
    id: int
    created_at: datetime

class MovementTransitionRequest(BaseModel):
    person_id: str
    semantic: Optional[str] = None # "entering" | "exiting" | "any"
    to_room_id: Optional[str] = None
    since_minutes: Optional[int] = None

class MovementTransitionResponse(BaseModel):
    person_id: str
    person_name: Optional[str] = None
    direction_semantic: str
    to_room_name: Optional[str] = None
    confidence: float
    observed_at: datetime

# --- Object Presence ---

class ObjectPresenceResponse(BaseModel):
    room_id: str
    object_label: str
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int
    last_observation_id: int
