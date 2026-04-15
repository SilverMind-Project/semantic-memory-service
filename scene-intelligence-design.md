# Scene Intelligence: Design & Implementation Plan

**Status**: Draft — 2026-04-13
**Scope**: Person tracking robustness, room transition inference, fast image analysis microservice, semantic scene memory microservice, cognitive-companion integration.

---

## 1. Problem Statement

The current system has four gaps this plan addresses:

1. **Direction is relative to the frame, not to the room.** The person-identification-service returns `"left-to-right"` but has no awareness that the camera is south-facing at the kitchen doorway, so left-to-right means "entering the kitchen". Without camera topology, downstream rules cannot express "person entered room" vs "person exited room".

2. **Scene context is ephemeral.** Vision model results (objects, descriptions) live only in `pipeline_data` for the duration of one pipeline run and are discarded. There is no way to ask "has there been a cardboard box near the stove for the last hour?" or "what was happening in the kitchen before this alert triggered?".

3. **Object & scene analysis is blocked on slow VLMs.** The `vision_analysis` / `llm_call` vision steps use large language-vision models (Cosmos-Reason2, Gemma4-26B multimodal) that take 30-60 s per clip. For fast, structured object detection ("what objects are in frame?", "is a stove burner on?") this is wasteful. A dedicated lightweight inference service can answer these questions in 1-3 s.

4. **Activity tracking is coarse-grained.** Activities are recorded after a heavy LLM decides the label, there is no temporal context accumulation, and the activity filter cannot express duration or sequence.

---

## 2. Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Edge                   New Microservices               Cognitive Companion     │
│  ─────                  ─────────────────               ──────────────────────  │
│                                                                                  │
│  reCamera ──►  Event   ──►  scene-analysis-service (8200)  ──►  pipeline steps    │
│               Aggregator    YOLO11 + Florence-2               (scene_analysis,  │
│               (CC)          Fast GPU inference                 person_id,       │
│                             < 2-3 s / image                    activity_detect) │
│                                    │                                │           │
│                                    │                                │           │
│                                    ▼                                ▼           │
│                         semantic-memory-service (8300)    Context Filters       │
│                         PostgreSQL + pgvector             (room_transition,     │
│                         Scene observations                 scene_contains,      │
│                         Person movements                   person_presence v2)  │
│                         CLIP embeddings                           │             │
│                         Temporal + vector queries                 │             │
│                                    ▲                              ▼             │
│                                    └──────── CC backend reads ────┘             │
│                                             (rules engine + filter eval)        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Service Summary

| Service | Port | Primary Role |
|---|---|---|
| person-identification-service | 8100 | Face recognition, direction vectors (existing) |
| **scene-analysis-service** | 8200 | YOLO11 object detection, Florence-2 scene description, CLIP embeddings |
| **semantic-memory-service** | 8300 | PostgreSQL+pgvector storage and retrieval of scene observations |

---

## 3. Camera Topology Configuration

### 3.1 Sensor Metadata Extension

Add to `Sensor.metadata_json` (no schema migration needed — existing JSON column):

```json
{
  "facing_direction_deg": 180,
  "camera_position": "doorway",
  "movement_map": {
    "left_to_right": {
      "semantic": "entering",
      "from_room_id": "hallway",
      "to_room_id": "kitchen"
    },
    "right_to_left": {
      "semantic": "exiting",
      "from_room_id": "kitchen",
      "to_room_id": "hallway"
    },
    "towards_camera": {
      "semantic": "approaching_exit",
      "from_room_id": "kitchen",
      "to_room_id": null
    },
    "away_from_camera": {
      "semantic": "entering_depth",
      "from_room_id": null,
      "to_room_id": "kitchen"
    }
  }
}
```

**`facing_direction_deg`**: 0 = north, 90 = east, 180 = south, 270 = west.

**`camera_position`**: `doorway` | `corner` | `ceiling` | `wall_mid`.

**`movement_map`**: Maps the person-identification-service's raw directions to semantically meaningful room transitions. Configured once per camera in the admin UI. The `from_room_id` / `to_room_id` reference rooms in the database.

### 3.2 Room Transition Inference

In `person_tracking.py`, when a camera event is processed:

```python
def _infer_room_transition(
    detection: PersonDetection,
    sensor: Sensor,
) -> RoomTransition | None:
    movement_map = sensor.metadata_json.get("movement_map", {})
    if not detection.direction:
        return None
    mapping = movement_map.get(detection.direction)
    if not mapping:
        return None
    return RoomTransition(
        semantic=mapping["semantic"],        # "entering" | "exiting" | ...
        from_room_id=mapping.get("from_room_id"),
        to_room_id=mapping.get("to_room_id"),
        confidence=detection.confidence,
        direction_raw=detection.direction,
    )
```

This `RoomTransition` is then recorded in `PersonLocationHistory` with `source="camera_topology"` and used to update `PersonLocationState`. The state update gains confidence because it has explicit semantic intent rather than "last-seen heuristic".

### 3.3 Admin UI

Add a camera topology panel to the sensor edit page:
- Facing direction (compass picker or degree input)
- Movement direction table (one row per raw direction, dropdowns for from/to room)
- Preview: renders "left-to-right → entering Kitchen from Hallway"

---

## 4. scene-analysis-service

### 4.1 Purpose

A stateless, GPU-accelerated inference service that answers two questions about an image in under 3 seconds:

1. **What objects are present and where?** (structured bounding-box output)
2. **What is the scene description?** (structured text, not free-form)

It does **not** attempt high-level reasoning or activity classification. That remains in the LLM steps of cognitive-companion for complex cases, or is rule-based within the service for simple cases (fire near stove = flag).

### 4.2 Model Selection

| Task | NVIDIA GPU | Intel GPU | Latency (approx) |
|---|---|---|---|
| Object detection | **YOLO11x** (PyTorch + CUDA) | **YOLO11x + OpenVINO** (INT8) | 30-80 ms / img |
| Scene description | **Florence-2-large** (PyTorch) | **Florence-2-base + IPEX** | 800 ms–1.5 s / img |
| CLIP embedding | **ViT-L/14** (OpenCLIP) | ViT-B/32 + OpenVINO | 50-100 ms / img |

**Why YOLO11x**: Best accuracy/speed tradeoff in the YOLO family. 500+ COCO classes. Runs at 5-30 FPS on a single consumer GPU. On a DGX Spark (A100 40GB) easily handles batches. On an Intel Arc A770, INT8 model runs at 15 FPS.

**Why Florence-2**: Microsoft's multi-task VFM at 232M params — technically a vision-language model but at a completely different scale than Gemma-4-26B. It produces structured outputs (JSON-like captions, region descriptions, grounded captions) not free-form prose. Inference is 500ms-1.5s per image vs 30-60s for the current vision step. For Intel, Florence-2-base (80M params) with IPEX runs in ~1s.

**Why not moondream2 / LLaVA-phi**: They're general chat VLMs. Florence-2 is purpose-built for structured visual tasks, which is exactly what we need.

**Why CLIP**: Produces dense image embeddings for vector search in semantic memory. OpenCLIP ViT-L/14 embeddings are 768-dim and semantically rich.

### 4.3 Directory Layout

```text
scene-analysis-service/
  app/
    main.py                # FastAPI app, lifespan handler
    config.py              # Settings (model paths, GPU selection, batch size)
    models/
      schemas.py           # Pydantic request/response models
    services/
      yolo_detector.py     # YOLO11 wrapper (NVIDIA + OpenVINO fallback)
      florence_describer.py # Florence-2 scene description
      clip_embedder.py     # CLIP embedding generation
      analyzer.py          # Combined analysis orchestrator
    routers/
      detect.py            # /api/v1/detect
      describe.py          # /api/v1/describe
      analyze.py           # /api/v1/analyze (detect + describe + embed)
      health.py
  Dockerfile
  pyproject.toml
  config/config.yaml
```

### 4.4 API

#### `POST /api/v1/detect`

```json
Request:
{
  "images": ["<base64>", ...],
  "confidence_threshold": 0.35,
  "classes": ["person", "fire", "knife", "stove"]   // optional filter
}

Response:
{
  "results": [
    {
      "frame_index": 0,
      "objects": [
        {
          "label": "person",
          "confidence": 0.92,
          "bbox": [x1, y1, x2, y2],
          "category": "person",
          "area_fraction": 0.18
        },
        {
          "label": "cardboard box",
          "confidence": 0.87,
          "bbox": [120, 300, 280, 450],
          "category": "container",
          "area_fraction": 0.04
        }
      ],
      "persons_count": 1,
      "hazard_flags": ["cardboard_near_stove"]
    }
  ],
  "processing_ms": 85
}
```

#### `POST /api/v1/describe`

```json
Request:
{
  "image": "<base64>",
  "tasks": ["caption", "region_descriptions", "object_list"]
}

Response:
{
  "caption": "A kitchen with a person near the stove. A cardboard box sits on the floor.",
  "region_descriptions": [
    {"region": [0, 0, 0.4, 1.0], "description": "person standing by stove"},
    {"region": [0.3, 0.6, 0.7, 1.0], "description": "cardboard box on floor"}
  ],
  "object_list": ["person", "stove", "cardboard box", "knife block", "bottles"],
  "processing_ms": 1200
}
```

#### `POST /api/v1/analyze` (primary endpoint)

```json
Request:
{
  "images": ["<base64>", ...],
  "include_description": true,
  "include_embedding": true,
  "sensor_id": "recamera_kitchen1"
}

Response:
{
  "results": [
    {
      "frame_index": 0,
      "objects": [...],          // from YOLO
      "persons_count": 1,
      "hazard_flags": ["cardboard_near_stove"],
      "description": "...",      // from Florence-2 (first frame only if multi-image)
      "object_list": [...],
      "embedding": [0.23, ...]   // CLIP 768-dim vector
    }
  ],
  "summary": {
    "unique_objects": ["person", "stove", "cardboard box"],
    "max_persons": 1,
    "any_hazards": true,
    "hazard_flags": ["cardboard_near_stove"]
  },
  "processing_ms": 1850
}
```

### 4.5 Hazard Rule Engine

Within the service, a lightweight rule engine maps detected objects + spatial relationships to `hazard_flags`:

```yaml
# config/hazards.yaml
rules:
  - name: cardboard_near_stove
    objects_required: [cardboard box, oven, stove]
    proximity_threshold: 0.3  # fraction of frame diagonal
    flag: cardboard_near_stove

  - name: unattended_burner
    objects_required: [stove, flame]
    absence_of: [person]
    flag: unattended_burner

  - name: trip_hazard_floor
    objects_required: [rug, mat, clothing]
    location_y_fraction_min: 0.6  # lower third of frame
    flag: trip_hazard_on_floor
```

These are fast spatial checks, not LLM reasoning. They produce structured flags that cognitive-companion rules can filter on directly.

### 4.6 GPU Device Selection

```python
# config.yaml
inference:
  device: auto   # "cuda", "openvino", "cpu", or "auto" (tries cuda > openvino > cpu)
  yolo_model: yolo11x.pt
  florence_model: microsoft/Florence-2-large
  clip_model: ViT-L/14
  batch_size: 4
  max_image_size: 1280  # resize before inference
```

`auto` mode probes for `torch.cuda.is_available()` → OpenVINO Core → CPU.

---

## 5. semantic-memory-service

### 5.1 Purpose

A queryable store of scene observations over time. It answers:

- "What objects have been detected in the kitchen in the last hour?"
- "Has Alice been seen entering the kitchen in the last 30 minutes?"
- "Find scenes similar to this image (vector search)."
- "What was the sequence of activities in the living room this morning?"
- "What objects were near the stove when the last hazard was detected?"

### 5.2 Technology Choice

**PostgreSQL + pgvector** over dedicated vector DBs (Qdrant, Milvus):
- Single persistence tier. The project already uses SQLite per service; Postgres unifies cognitive-companion's eventual migration path.
- pgvector supports IVFFlat and HNSW indexes — sufficient for thousands to millions of observations at home scale.
- Rich temporal queries (timestamp ranges, INTERVAL arithmetic) that vector-only DBs can't do natively.
- `pg_partitioning` by day for cheap time-based pruning.
- Runs as a Docker container; same operational footprint.

### 5.3 Schema

```sql
-- Core scene observation
CREATE TABLE scene_observations (
    id              BIGSERIAL PRIMARY KEY,
    sensor_id       TEXT NOT NULL,
    room_id         TEXT,
    room_name       TEXT,
    observed_at     TIMESTAMPTZ NOT NULL,
    source          TEXT NOT NULL,  -- 'scene_intel' | 'llm_vision' | 'manual'

    -- Structured detection output
    objects_json    JSONB,          -- [{label, confidence, bbox, category}]
    persons_count   INT,
    hazard_flags    TEXT[],
    description     TEXT,
    object_list     TEXT[],         -- denormalized for fast array ops

    -- Raw pipeline context (optional, for auditability)
    workflow_execution_id  BIGINT,
    media_paths_json       JSONB,

    -- Vector embedding (CLIP ViT-L/14 = 768 dims)
    embedding       vector(768),

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scene_obs_sensor_time   ON scene_observations (sensor_id, observed_at DESC);
CREATE INDEX idx_scene_obs_room_time     ON scene_observations (room_id, observed_at DESC);
CREATE INDEX idx_scene_obs_hazards       ON scene_observations USING GIN (hazard_flags);
CREATE INDEX idx_scene_obs_objects       ON scene_observations USING GIN (object_list);
CREATE INDEX idx_scene_obs_embedding     ON scene_observations USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Person movement events (replaces PersonLocationHistory in CC for camera-sourced events)
CREATE TABLE person_movements (
    id              BIGSERIAL PRIMARY KEY,
    person_id       TEXT NOT NULL,
    person_name     TEXT,

    sensor_id       TEXT NOT NULL,
    from_room_id    TEXT,
    to_room_id      TEXT,
    from_room_name  TEXT,
    to_room_name    TEXT,

    direction_raw   TEXT,           -- "left-to-right" etc.
    direction_semantic TEXT,        -- "entering" | "exiting" | "approaching_exit" | "stationary"

    confidence      FLOAT,
    observed_at     TIMESTAMPTZ NOT NULL,
    observation_id  BIGINT REFERENCES scene_observations(id),

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_person_movements_person_time ON person_movements (person_id, observed_at DESC);
CREATE INDEX idx_person_movements_room_time   ON person_movements (to_room_id, observed_at DESC);

-- Timeline materialized view: latest location per person
CREATE MATERIALIZED VIEW person_current_location AS
SELECT DISTINCT ON (person_id)
    person_id, person_name, to_room_id AS room_id, to_room_name AS room_name,
    direction_semantic, confidence, observed_at
FROM person_movements
WHERE to_room_id IS NOT NULL
ORDER BY person_id, observed_at DESC;

CREATE UNIQUE INDEX ON person_current_location (person_id);

-- Object timeline: latest occurrence of each object per room (for context filters)
CREATE TABLE object_presence (
    id              BIGSERIAL PRIMARY KEY,
    room_id         TEXT NOT NULL,
    object_label    TEXT NOT NULL,
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_seen_at    TIMESTAMPTZ NOT NULL,
    observation_count INT DEFAULT 1,
    last_observation_id BIGINT REFERENCES scene_observations(id)
);

CREATE UNIQUE INDEX ON object_presence (room_id, object_label);
```

### 5.4 API

#### `POST /api/v1/observations`

```json
Request:
{
  "sensor_id": "recamera_kitchen1",
  "room_id": "kitchen",
  "room_name": "Kitchen",
  "observed_at": "2026-04-13T01:35:28Z",
  "source": "scene_intel",
  "objects": [...],
  "persons_count": 1,
  "hazard_flags": ["cardboard_near_stove"],
  "description": "A kitchen with a person near the stove...",
  "object_list": ["person", "stove", "cardboard box"],
  "embedding": [0.23, ...],
  "workflow_execution_id": 42,
  "media_paths": [...]
}

Response:
{ "id": 10291, "created_at": "..." }
```

#### `POST /api/v1/observations/search`

```json
Request:
{
  "query_embedding": [0.23, ...],   // optional: vector similarity
  "room_id": "kitchen",             // optional: room filter
  "since_minutes": 60,              // optional: time window
  "objects_any": ["fire", "flame"], // optional: must contain any
  "hazard_flags_any": ["unattended_burner"],  // optional
  "limit": 20,
  "similarity_threshold": 0.75      // minimum cosine similarity
}

Response:
{
  "results": [
    {
      "id": 10290,
      "observed_at": "...",
      "room_name": "Kitchen",
      "similarity": 0.89,
      "description": "...",
      "hazard_flags": [],
      "object_list": [...]
    }
  ]
}
```

#### `POST /api/v1/movements`

```json
Request:
{
  "person_id": "person1",
  "person_name": "Alice",
  "sensor_id": "recamera_kitchen1",
  "from_room_id": "hallway",
  "to_room_id": "kitchen",
  "direction_raw": "left-to-right",
  "direction_semantic": "entering",
  "confidence": 0.87,
  "observed_at": "2026-04-13T01:35:28Z",
  "observation_id": 10291
}
```

#### `GET /api/v1/movements/transitions`

```
Query params:
  person_id=person1
  semantic=entering
  to_room_id=kitchen
  since_minutes=30

Response:
{
  "transitions": [
    {
      "person_id": "person1",
      "person_name": "Alice",
      "direction_semantic": "entering",
      "to_room_name": "Kitchen",
      "observed_at": "...",
      "confidence": 0.87
    }
  ]
}
```

#### `GET /api/v1/timeline/{person_id}`

Returns movement timeline for a person, optionally filtered by time range.

#### `GET /api/v1/objects/{room_id}/recent`

```
Query params: since_minutes=60, min_confidence=0.5

Response:
{
  "room_id": "kitchen",
  "objects": [
    {"label": "cardboard box", "last_seen_minutes_ago": 12, "observation_count": 4},
    {"label": "person", "last_seen_minutes_ago": 1, "observation_count": 28}
  ]
}
```

#### `GET /health`, `DELETE /api/v1/observations/prune` (retention management)

### 5.5 Directory Layout

```text
semantic-memory-service/
  app/
    main.py
    config.py
    db/
      connection.py         # asyncpg connection pool
      migrations/           # SQL migration scripts
        001_initial.sql
    models/
      schemas.py            # Pydantic request/response models
    services/
      observation_store.py  # Insert + query scene_observations
      movement_store.py     # Insert + query person_movements
      object_presence.py    # Upsert object_presence aggregates
      search.py             # Vector + temporal + filter queries
    routers/
      observations.py
      movements.py
      objects.py
      health.py
  Dockerfile
  pyproject.toml
  config/config.yaml
```

---

## 6. Cognitive Companion Integration

### 6.1 New Integrations

```text
backend/integrations/
  scene_intel_client.py    # HTTP client for scene-analysis-service
  semantic_memory_client.py # HTTP client for semantic-memory-service
```

Both follow the same pattern as `person_id_client.py`: async httpx, graceful degradation when unconfigured, typed return dataclasses.

### 6.2 New Pipeline Steps

#### `scene_analysis` step

**Type**: `"scene_analysis"` | **Category**: `"perception"`

**Purpose**: Call scene-analysis-service on the trigger's media_paths. Fast (2-3s). Runs before or in parallel with person_identification.

**Config Schema**:
```json
{
  "include_description": true,
  "include_embedding": true,
  "classes_filter": [],           // limit YOLO to these classes (empty = all)
  "confidence_threshold": 0.35,
  "write_to_memory": true,        // auto-write result to semantic-memory-service
  "additional_sensor_ids": []     // same cross-camera aggregation as person_id step
}
```

**Output** (`pipeline_data`):
```json
{
  "scene_analysis": {
    "objects": [{"label": "cardboard box", "confidence": 0.87, "bbox": [...]}],
    "object_list": ["person", "cardboard box", "stove"],
    "persons_count": 1,
    "hazard_flags": ["cardboard_near_stove"],
    "description": "A kitchen with a person near the stove...",
    "observation_id": 10291
  }
}
```

The `observation_id` links this pipeline execution to the semantic memory record, enabling auditability.

#### `semantic_memory_write` step

**Type**: `"semantic_memory_write"` | **Category**: `"state"`

**Purpose**: Explicitly write any pipeline_data fields to semantic memory (for cases where the scene_analysis step isn't used but you want to persist LLM results).

**Config Schema**:
```json
{
  "description_key": "vision_response",  // pipeline_data key to use as description
  "objects_key": "scene_analysis.object_list",  // dot-path supported
  "hazard_flags_key": "scene_analysis.hazard_flags"
}
```

#### `semantic_memory_query` step

**Type**: `"semantic_memory_query"` | **Category**: `"context"`

**Purpose**: Query semantic memory and inject results into pipeline_data for use in downstream LLM prompts.

**Config Schema**:
```json
{
  "output_key": "memory_context",
  "room_id": "{{trigger.room_id}}",
  "since_minutes": 60,
  "objects_any": [],
  "hazard_flags_any": [],
  "limit": 5,
  "format": "summary"   // "summary" | "full" | "object_list"
}
```

**Output**:
```json
{
  "memory_context": {
    "recent_objects": ["cardboard box", "stove", "person"],
    "recent_hazards": ["cardboard_near_stove"],
    "summary": "In the past hour: cardboard box detected near stove (3 times), 1 person present.",
    "observations_count": 4
  }
}
```

This enables LLM prompts like:
> `"Context from the last hour: {{memory_context.summary}}. Based on the current image, assess..."`

#### Enhanced `person_identification` step

Add to existing config schema:

```json
{
  "infer_room_transitions": true,   // use camera topology to compute entering/exiting
  "write_movements_to_memory": true // push transitions to semantic-memory-service
}
```

**Enhanced output**:
```json
{
  "person_detections": [...],
  "room_transitions": [
    {
      "person_id": "person1",
      "person_name": "Alice",
      "semantic": "entering",
      "from_room_name": "Hallway",
      "to_room_name": "Kitchen",
      "confidence": 0.87,
      "direction_raw": "left-to-right"
    }
  ]
}
```

### 6.3 New Context Filters

#### `room_transition` filter

**Type**: `"room_transition"`

**Purpose**: Gate a rule on a person having entered or exited a specific room within a time window. More reliable than `person_presence` because it uses explicit movement events, not just "last seen".

**Config**:
```json
{
  "person_id": "person1",
  "semantic": "entering",     // "entering" | "exiting" | "any"
  "room_id": "kitchen",
  "within_minutes": 5
}
```

**Evaluation**: Queries `semantic-memory-service GET /api/v1/movements/transitions`.

**Use cases**:
- "Alert only when Mom enters the kitchen (not when she's already been there)"
- "Send reminder when Dad exits the bedroom in the morning"
- "Trigger medication check if person enters bathroom"

#### `scene_contains` filter

**Type**: `"scene_contains"`

**Purpose**: Gate a rule on an object or hazard being present in a room recently.

**Config**:
```json
{
  "room_id": "kitchen",
  "objects_any": ["cardboard box", "clothing"],   // OR
  "hazard_flags_any": ["cardboard_near_stove"],   // OR
  "within_minutes": 30
}
```

**Evaluation**: Queries `semantic-memory-service GET /api/v1/objects/{room_id}/recent`.

**Use cases**:
- "Alert only if the hazard has been present for more than 15 minutes (persistent)"
- "Skip reminder if stove is not currently on"

#### Enhanced `person_presence` v2

Extends the current filter with semantic memory fallback:

```json
{
  "person_id": "person1",
  "status": "home",
  "room_name": "Kitchen",
  "within_minutes": 15,       // new: tighter configurable window (replaces hardcoded 30)
  "use_semantic_memory": true  // new: also check semantic-memory-service movements
}
```

When `use_semantic_memory=true`, evaluation also checks `person_current_location` materialized view in semantic memory, which is maintained by movement events and is more accurate than the camera-sighting heuristic currently used.

### 6.4 Settings YAML Changes

```yaml
# config/settings.yaml additions

scene_intel:
  url: http://scene-analysis-service:8200
  enabled: true
  timeout: 10

semantic_memory:
  url: http://semantic-memory-service:8300
  enabled: true
  timeout: 5
  retention_days: 90        # auto-prune observations older than this
```

### 6.5 ServiceContainer Extension

```python
# steps/base.py — ServiceContainer additions
@dataclass
class ServiceContainer:
    # ... existing fields ...
    scene_intel_client: SceneIntelClient | None = None
    semantic_memory_client: SemanticMemoryClient | None = None
```

Both clients are initialized in `main.py` alongside `person_id_client` and injected into pipelines.

---

## 7. Person Activity Tracking Next Generation

### 7.1 Current Limitations

- Activity is a point-in-time label, no duration.
- No temporal sequences ("person ate, then went to bedroom → sleep hypothesis").
- Activity filter only checks "did this activity occur in last N minutes", not "is activity ongoing".
- Scene description is stored as a raw string blob.

### 7.2 Activity Enrichment via Scene Intel

The `activity_detection` step gains a new optional field:

```json
{
  "enrich_from_scene": true,           // pull object_list from scene_analysis step output
  "scene_analysis_key": "scene_analysis"  // pipeline_data key
}
```

When enabled, `metadata_json` for the activity record includes:
```json
{
  "scene_description": "...",
  "objects_in_scene": ["stove", "plate", "person"],
  "hazard_flags": [],
  "observation_id": 10291           // backlink to semantic memory
}
```

### 7.3 Activity Sequences (new concept)

A new table in cognitive-companion's SQLite (or in semantic-memory-service's Postgres):

```sql
CREATE TABLE activity_sequences (
    id              BIGSERIAL PRIMARY KEY,
    person_id       TEXT NOT NULL,
    sequence_type   TEXT NOT NULL,  -- e.g., "morning_routine", "medication_cycle"
    activities_json JSONB,          -- ordered list of activity records
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    confidence      FLOAT,
    status          TEXT            -- "in_progress" | "completed" | "broken"
);
```

Rules can define expected sequences. A future `sequence_detection` step would close open sequences or trigger alerts when sequences break ("medication sequence started but no take_medication in 30 min").

### 7.4 `person_activity` Filter v2

```json
{
  "person_id": "person1",
  "activity_type": "eating",
  "within_minutes": 30,
  "min_confidence": 0.7,    // new: filter low-confidence activities
  "status": "completed",    // new: "completed" | "in_progress" | "any"
  "include_semantic": true  // new: check object presence in room as corroboration
}
```

---

## 8. Implementation Sequence

The plan is designed so each phase is independently shippable and doesn't break existing functionality.

### Phase 1: Camera Topology (2-3 days)
No new services. Pure cognitive-companion work.

1. Add `movement_map` to Sensor metadata schema (documented field, no migration).
2. Implement `_infer_room_transition()` in `person_tracking.py`.
3. Update `PersonLocationHistory` writes to include `direction_semantic` and `from_room_id` (add columns via Alembic migration).
4. Expose `movement_map` editor in sensor admin UI.
5. Update `person_identification` step to output `room_transitions` alongside `person_detections`.
6. Add `room_transition` context filter reading from `PersonLocationHistory`.
7. Tests: topology inference, location state updates, filter evaluation.

**Deliverable**: Rules can now express "alert when person enters kitchen" using existing DB.

---

### Phase 2: scene-analysis-service (3-5 days)

1. Scaffold service: FastAPI + Dockerfile + pyproject.toml.
2. Implement `YOLODetector` with CUDA/OpenVINO auto-selection.
3. Implement `FlorenceDescriber` (Florence-2-large, INT8 quantization for speed).
4. Implement `ClipEmbedder` (OpenCLIP ViT-L/14).
5. Implement hazard rule engine (YAML config, spatial proximity checks).
6. Wire `/api/v1/analyze` endpoint.
7. Add `SceneIntelClient` to cognitive-companion.
8. Add `scene_analysis` pipeline step.
9. Tests: detector mock, hazard rule engine, client graceful degradation.
10. Docker Compose entry.

**Deliverable**: `scene_analysis` step available in pipelines, returns structured objects + hazards in ~2s.

---

### Phase 3: semantic-memory-service (4-6 days)

1. Scaffold service with asyncpg + pgvector.
2. Write SQL migrations (001_initial.sql).
3. Implement observation store, movement store, object presence upsert.
4. Implement HNSW vector search + temporal + filter queries.
5. Add `SemanticMemoryClient` to cognitive-companion.
6. Wire auto-write from `scene_analysis` step (`write_to_memory: true`).
7. Wire auto-write from `person_identification` step (`write_movements_to_memory: true`).
8. Add `semantic_memory_write` and `semantic_memory_query` pipeline steps.
9. Add `scene_contains` context filter.
10. Enhance `person_presence` filter with semantic memory fallback.
11. Tests: store/retrieve, vector search, filter evaluation with mock client.
12. Docker Compose entry.

**Deliverable**: Scene history queryable in filters and LLM prompt context.

---

### Phase 4: Activity Enrichment (1-2 days)

1. Add `enrich_from_scene` option to `activity_detection` step.
2. Add `observation_id` backlink to `PersonActivity.metadata_json`.
3. Update `person_activity` filter with `min_confidence` and `include_semantic` options.
4. Tests.

**Deliverable**: Activities have richer metadata; filter is more configurable.

---

## 9. Non-Goals (explicitly out of scope)

- **Pose estimation**: Human pose (sitting, standing, lying) adds significant model complexity. A future phase using DWPose or ViTPose on NVIDIA.
- **Audio integration**: Speech activity detection (cooking sounds, TV on) — separate effort.
- **Federated learning / embedding updates**: The enrollment model update cycle remains manual.
- **Real-time streaming inference**: The service is request/response. Streaming video analysis is a separate architectural decision.
- **Re-identification without enrollment**: Cross-camera tracking of unknowns requires a separate re-ID model (OSNet, BoT-SORT). The current unknown_0/unknown_1 synthetic tracking is acceptable for now.

---

## 10. Key Design Invariants

1. **Cognitive-companion is always the orchestrator.** The new microservices are dumb inference/storage engines. All pipeline logic, rule evaluation, and alert routing stays in CC.

2. **Graceful degradation everywhere.** If `scene-analysis-service` is down, `scene_analysis` step returns empty results and sets `should_continue=False` only if the rule requires it. The existing LLM path remains untouched.

3. **No new database in CC.** CC's SQLite stays authoritative for rules, rooms, persons, and alert state. `semantic-memory-service` is a read-optimised projection store, not the source of truth for person location (that remains `PersonLocationState`).

4. **Observation IDs link everything.** `scene_observations.id` ties together: the CC workflow execution, the scene-intel result, the person movements, and the activity record. Full auditability across services.

5. **camera topology is per-sensor configuration, not code.** The movement_map is data, not logic. A new camera is configured in the admin UI without a code change.
