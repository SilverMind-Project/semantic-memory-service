# AGENTS.md

Reference for AI coding agents working in `semantic-memory-service/`. This document is the canonical, deep guide. `CLAUDE.md` is a tight pointer aimed at the same audience; `README.md` is human-facing.

If a fact appears here, it traces to a file in this tree at the time of writing. Verify before relying on it: `git log` is authoritative for "what changed", and `grep` against `app/` is authoritative for "what exists".

---

## 1. Mission and scope

Semantic Memory Service is a write-through, query-heavy microservice that provides time-series searchable memory for the Cognitive Companion system. It stores:
- **Scene observations**: structured descriptions of what was seen in a room, with CLIP image embeddings and text embeddings for semantic search
- **Person movement transitions**: room-to-room movements with semantic direction (entering, exiting, approaching_exit, entering_depth, stationary)
- **Object presence state**: aggregated "last seen" tracking of objects in rooms via upsert semantics

Cognitive Companion is the sole consumer. It writes observations and movements through the API, and reads them back via vector similarity search, temporal queries, and transition lookups. MCP tools and pipeline steps (`semantic_memory_query`, `semantic_memory_write`, `object_trend_analysis`) query this service to inform LLM reasoning and rule evaluation.

---

## 2. Tech stack

| Layer | Choice |
| --- | --- |
| Runtime | Python 3.14 |
| Framework | FastAPI (async) |
| Database driver | psycopg3 (`psycopg[binary,pool]`) with `AsyncConnectionPool` |
| Database | PostgreSQL 18 via `timescale/timescaledb-ha:pg18` (shared instance), `semantic_memory` database |
| Vector index | pgvectorscale StreamingDiskANN indexes |
| Text embeddings | embeddinggemma-300m via Triton Inference Server (gRPC), wrapped by `triton-shared` library |
| Migrations | Alembic (raw SQL, no ORM models) |
| Package manager | `uv` (`uv.lock` is committed) |
| Lint / types | `ruff`; `mypy` (not strict) |
| Tests | `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`) |

---

## 3. Repository layout

```text
semantic-memory-service/
├── app/
│   ├── main.py                    FastAPI app factory, lifespan (DB connect, migrations, router registration)
│   ├── run.py                     Standalone server entry point (port 8400)
│   ├── config/
│   │   └── config.py              Settings (pydantic-settings, .env)
│   ├── db/
│   │   ├── connection.py           AsyncConnectionPool singleton
│   │   ├── migrate.py              Alembic migration runner
│   │   └── alembic/
│   │       ├── env.py              Alembic environment (async engine from DATABASE_URL)
│   │       ├── script.py.mako      Migration template
│   │       └── versions/
│   │           └── 0001_initial_schema.py   Current schema
│   ├── models/
│   │   └── schemas.py             Pydantic wire models (ObservationCreate, MovementCreate, SearchRequest, etc.)
│   ├── routers/
│   │   ├── observations.py        POST /observations, POST /observations/search, DELETE /observations/prune
│   │   ├── movements.py           POST /movements, GET /movements/transitions
│   │   └── objects.py             GET /objects/{room_id}/recent
│   └── services/
│       ├── observation_store.py   ObservationStore: create, get_by_id
│       ├── movement_store.py      MovementStore: create, get_transitions
│       ├── search.py              SearchService: vector similarity + metadata filter search
│       ├── object_presence.py     ObjectPresenceStore: upsert_presence, get_by_room, delete_old_records
│       └── text_embedder.py       TextEmbedder ABC, TritonTextEmbedder, NullTextEmbedder, build_text_embedder()
├── config/                        (currently empty; reserved for future config files)
├── tests/
│   ├── test_services.py           Unit tests for all stores, search, and text embedder
│   └── test_api.py                HTTP integration tests for all endpoints
├── pyproject.toml                 Dependencies and tool config
├── uv.lock                        Committed lock file
├── Dockerfile                     3.14-slim, port 8400, tokenizer download at build time
├── docker-compose.yml             Single service, external nanai network
├── alembic.ini                    Alembic config pointing to app/db/alembic
├── .env                           Local dev environment (DATABASE_URL, SMS_DB_USER, SMS_DB_PASSWORD)
└── LICENSE                        AGPL-3.0
```

---

## 4. Commands

Run from the repository root unless noted.

```bash
# Development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8400 --reload

# Production (Docker)
docker compose up --build           # API at http://localhost:8400

# Tests
uv run pytest                       # full suite
uv run pytest -v                    # verbose
uv run pytest tests/test_services.py -v
uv run pytest tests/test_api.py -v

# Lint and types
uv run ruff check .                 # lint
uv run ruff check --fix .           # auto-fix
uv run mypy app/ --ignore-missing-imports --explicit-package-bases

# Database migrations
uv run alembic revision -m "description of change"
uv run alembic upgrade head
uv run alembic downgrade -1
```

---

## 5. Architecture

### 5.1 Layering

```text
app/routers/          HTTP endpoints: request parsing, response formatting, HTTP error mapping
      │
app/services/          Business logic: stores (create, query), search (vector + filter), text embedder
      │
app/db/connection.py   AsyncConnectionPool (psycopg3)
      │
PostgreSQL             scene_observations, person_movements, object_presence
                       + pgvectorscale StreamingDiskANN indexes
                       + person_current_location materialized view
```

### 5.2 Lifespan

```text
startup  ─►  db.connect()  ─►  run_migrations() (alembic upgrade head via asyncio.to_thread)
    │
serve   (all requests)
    │
shutdown ─►  db.disconnect()  (close pool, set to None)
```

`app/main.py` creates the `FastAPI` app with `lifespan=lifespan`. `app/run.py` is a standalone entry point that manages its own `db.connect()` / `db.disconnect()` for running via `python -m app.run` (used by Dockerfile CMD).

### 5.3 Database connection

```python
from app.db.connection import db

# In services: get the pool and borrow a connection
pool = db.get_pool()
async with pool.connection() as conn:
    async with conn.cursor() as cur:
        await cur.execute("SELECT ...", params)
        rows = await cur.fetchall()
```

All queries use raw SQL via `psycopg` async cursors. There are no SQLAlchemy ORM models, no `session.execute()`, no `select()` builders. The only SQLAlchemy usage is Alembic (for migration tooling) and the `create_async_engine` in `env.py`.

### 5.4 Configuration

```python
from app.config.config import settings

url = settings.DATABASE_URL        # Must be set via env
enabled = settings.TEXT_EMBEDDING_ENABLED  # default True
```

Settings are loaded from `.env` via `pydantic-settings`. No YAML config files. The `config/` directory is empty (reserved for future use).

| Setting | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | (required) | PostgreSQL connection string |
| `API_V1_STR` | `/api/v1` | API version prefix |
| `PROJECT_NAME` | `semantic-memory-service` | OpenAPI doc title |
| `RETENTION_DAYS` | `90` | Default prune window |
| `TEXT_EMBEDDING_ENABLED` | `true` | Enable Triton text embeddings |
| `TRITON_URL` | `localhost:8701` | Triton gRPC endpoint |
| `TRITON_TEXT_EMBEDDING_MODEL` | `embeddinggemma-300m` | Triton model name |
| `TRITON_TEXT_EMBEDDING_TOKENIZER_PATH` | `/models/embeddinggemma-300m/1/tokenizer.json` | Path to tokenizer.json |

---

## 6. API reference

Base path: `/api/v1`

### 6.1 Observations

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/observations/` | Create an observation. Returns `201` with `id` and `created_at`. |
| `POST` | `/observations/search` | Vector similarity search with metadata filters. |
| `DELETE` | `/observations/prune?days=N` | Delete observations older than N days (default: `RETENTION_DAYS`). |

**ObservationCreate fields:**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `sensor_id` | `str \| None` | No | Source sensor |
| `room_id` | `str \| None` | No | Room identifier |
| `room_name` | `str \| None` | No | Human-readable room name |
| `observed_at` | `datetime` | Yes | When the observation was made |
| `source` | `str` | Yes | `scene_intel`, `llm_vision`, or `manual` |
| `objects_json` | `list[dict] \| None` | No | Full object detection data |
| `persons_count` | `int \| None` | No | Count of persons detected |
| `hazard_flags` | `list[str] \| None` | No | Hazard labels (e.g. `["fire", "weapon"]`) |
| `description` | `str \| None` | No | Natural language description |
| `description_embedding` | `list[float] \| None` | No | 768-dim text embedding |
| `object_list` | `list[str] \| None` | No | Flat list of object labels |
| `workflow_execution_id` | `int \| None` | No | Correlated CC workflow execution |
| `media_paths_json` | `list[str] \| None` | No | MinIO paths for associated media |
| `embedding` | `list[float] \| None` | No | 768-dim CLIP image embedding |

**ObservationSearchRequest fields:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `query_embedding` | `list[float] \| None` | None | CLIP embedding for image similarity |
| `query_text` | `str \| None` | None | Text for semantic search via Triton |
| `room_id` | `str \| None` | None | Filter by room |
| `since_minutes` | `int \| None` | None | Time window filter |
| `objects_any` | `list[str] \| None` | None | Array overlap filter (`&&`) |
| `hazard_flags_any` | `list[str] \| None` | None | Array overlap filter (`&&`) |
| `limit` | `int` | 20 | Max results |
| `similarity_threshold` | `float` | 0.75 | Cosine similarity floor (0-1) |

Search uses `<=>` (cosine distance) with pgvectorscale StreamingDiskANN indexes. When both `query_embedding` and `query_text` are provided, results are ordered by the average of the two similarity scores.

### 6.2 Movements

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/movements/` | Create a movement record. Returns `201` with `id` and `created_at`. |
| `GET` | `/movements/transitions?person_id=&semantic=&to_room_id=&since_minutes=` | Query movement transitions. |

**MovementCreate fields:**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `person_id` | `str` | Yes | Person identifier |
| `person_name` | `str \| None` | No | Human-readable name |
| `sensor_id` | `str \| None` | No | Camera that captured the movement |
| `from_room_id` | `str \| None` | No | Source room |
| `to_room_id` | `str \| None` | No | Destination room |
| `from_room_name` | `str \| None` | No | Source room display name |
| `to_room_name` | `str \| None` | No | Destination room display name |
| `direction_raw` | `str \| None` | No | Raw direction from camera (e.g. `left_to_right`) |
| `direction_semantic` | `str` | Yes | Semantically mapped direction: `entering`, `exiting`, `approaching_exit`, `entering_depth`, `stationary` |
| `confidence` | `float` | Yes | Detection confidence (0-1) |
| `observed_at` | `datetime` | Yes | When the movement was observed |
| `observation_id` | `int \| None` | No | FK to `scene_observations.id` |

**Transition query parameters:**

| Parameter | Type | Required | Notes |
| --- | --- | --- | --- |
| `person_id` | `str` | Yes | Person to query |
| `semantic` | `str \| None` | No | Filter by `direction_semantic` |
| `to_room_id` | `str \| None` | No | Filter by destination room |
| `since_minutes` | `int \| None` | No | Time window |

### 6.3 Objects

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/objects/{room_id}/recent?since_minutes=60` | Get recent object presence in a room. |

Response includes `label`, `last_seen_minutes_ago`, and `observation_count` for each object.

### 6.4 Health

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Returns `{"status": "healthy", "service": "semantic-memory-service"}`. |

---

## 7. Database schema

Three tables and one materialized view. All timestamps are `TIMESTAMPTZ`. Vector columns use `vector(768)` from pgvectorscale.

### 7.1 scene_observations

```sql
id                      BIGSERIAL PRIMARY KEY
sensor_id               TEXT NOT NULL
room_id                 TEXT
room_name               TEXT
observed_at             TIMESTAMPTZ NOT NULL
source                  TEXT NOT NULL            -- 'scene_intel', 'llm_vision', 'manual'
objects_json            JSONB
persons_count           INT
hazard_flags            TEXT[]                   -- GIN-indexed
description             TEXT
description_embedding   vector(768)              -- DiskANN-indexed
object_list             TEXT[]                   -- GIN-indexed
workflow_execution_id   BIGINT
media_paths_json        JSONB
embedding               vector(768)              -- DiskANN-indexed (CLIP)
created_at              TIMESTAMPTZ DEFAULT NOW()
```

Indexes:
- B-tree on `(sensor_id, observed_at DESC)`
- B-tree on `(room_id, observed_at DESC)`
- GIN on `hazard_flags`
- GIN on `object_list`
- **StreamingDiskANN** on `embedding`
- **StreamingDiskANN** on `description_embedding`

### 7.2 person_movements

```sql
id                      BIGSERIAL PRIMARY KEY
person_id               TEXT NOT NULL
person_name             TEXT
sensor_id               TEXT NOT NULL
from_room_id            TEXT
to_room_id              TEXT
from_room_name          TEXT
to_room_name            TEXT
direction_raw           TEXT
direction_semantic      TEXT
confidence              FLOAT
observed_at             TIMESTAMPTZ NOT NULL
observation_id          BIGINT REFERENCES scene_observations(id)
created_at              TIMESTAMPTZ DEFAULT NOW()
```

Indexes:
- B-tree on `(person_id, observed_at DESC)`
- B-tree on `(to_room_id, observed_at DESC)`

Materialized view `person_current_location`: `SELECT DISTINCT ON (person_id) ... ORDER BY person_id, observed_at DESC`, with a unique index on `person_id`.

### 7.3 object_presence

```sql
id                      BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY
room_id                 TEXT NOT NULL
object_label            TEXT NOT NULL
first_seen_at           TIMESTAMPTZ NOT NULL
last_seen_at            TIMESTAMPTZ NOT NULL
observation_count       INT DEFAULT 1
last_observation_id     BIGINT REFERENCES scene_observations(id)
```

Unique constraint on `(room_id, object_label)`. Insert uses `ON CONFLICT ... DO UPDATE` upsert semantics: `last_seen_at`, `observation_count`, and `last_observation_id` are updated on conflict.

---

## 8. Text embedding subsystem

```
TextEmbedder (ABC)
    ├── TritonTextEmbedder   wraps triton-shared, lazy gRPC init, 768-dim
    └── NullTextEmbedder     returns [], is_available=False

build_text_embedder(enabled, triton_url, model_name, tokenizer_path) -> TextEmbedder
```

The `SearchService` uses the embedder to convert `query_text` into a 768-dim embedding for cosine similarity search against `description_embedding`. If text embedding is disabled or Triton is unreachable, the `NullTextEmbedder` returns `[]` and the search falls back to metadata-only (no text similarity score).

**Lazy initialization**: `TritonTextEmbedder._ensure_client()` is called on first `embed()`. It creates a `TritonGrpcClient`, enters the async context, and wraps it in a `triton_shared.models.embedder.TextEmbedder`. This avoids blocking startup if Triton is not yet ready.

**Docker build**: The tokenizer.json is downloaded at build time from HuggingFace to `/models/embeddinggemma-300m/1/tokenizer.json`. This path must match `TRITON_TEXT_EMBEDDING_TOKENIZER_PATH`.

---

## 9. Error handling

Each store/service defines its own exception class:
- `ObservationStoreError` (400)
- `MovementStoreError` (400)
- `SearchServiceError` (500)
- `ObjectPresenceStoreError` (400)

Routers catch these and map them to HTTP responses. The global `service_exception_handler` in `app/main.py` catches unhandled instances from any router path.

Exceptions never bubble out of integration code. The `NullTextEmbedder` returns `[]` instead of raising. The `SearchService` returns empty lists on DB errors (wrapped in `SearchServiceError`).

---

## 10. Testing conventions

### 10.1 Pattern

Tests use `unittest.mock` for all infrastructure dependencies. No testcontainers, no real database. The database pool is patched at `app.db.connection.db.get_pool` with a mock pool that returns mock connections with mock cursors.

```python
def _make_mock_cursor(return_value=None, fetchall_return=None, fetchone_return=None, description=None):
    cur = MagicMock()
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(return_value=fetchone_return)
    cur.fetchall = AsyncMock(return_value=fetchall_return or [])
    cur.description = description or []
    return cur

def _mock_db_pool(monkeypatch, cursor):
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=conn)
    monkeypatch.setattr("app.db.connection.db.get_pool", lambda: mock_pool)
    return mock_pool
```

### 10.2 API tests

API tests use `fastapi.testclient.TestClient` or `httpx.AsyncClient` with `ASGITransport`. The `setup_db` fixture patches `db.pool`, `db.connect()`, `db.disconnect()`, `run_migrations()`, and the text embedder to `NullTextEmbedder` before each test.

### 10.3 What to test

- Store success path: create returns expected `id`/`created_at`, query returns expected rows
- Store failure path: DB error raises the correct store error
- Search: filter combinations, similarity score inclusion, empty results, error wrapping
- Text embedder: `NullTextEmbedder` returns `[]`, `TritonTextEmbedder` properties, `build_text_embedder` factory logic
- API: 201 on create, 200 on search/transitions, 422 on invalid payload, 404 on unknown routes

### 10.4 What NOT to do

- Do not use a real database in tests. Mock the pool and cursor.
- Do not test Alembic migrations in unit tests. Schema shape is verified by code review of migration files.
- Do not add testcontainers. The service is stateless at the Python level; all state is in PostgreSQL.
- Do not test the Triton client. The `TritonTextEmbedder` uses triton-shared which is tested upstream.

---

## 11. Common tasks

### 11.1 Add a new API endpoint

1. Add Pydantic request/response schemas to `app/models/schemas.py`.
2. Create or extend a service in `app/services/`.
3. Create or extend a router in `app/routers/`.
4. Register the router in `app/main.py` (`app.include_router`).
5. Add tests to `tests/test_api.py` and `tests/test_services.py`.

### 11.2 Add a new database table

1. Create a new Alembic migration: `uv run alembic revision -m "add <table>"`.
2. Write `op.execute("CREATE TABLE ...")` with raw SQL.
3. Add appropriate indexes (B-tree for time ranges, GIN for arrays, DiskANN for vectors).
4. Write the corresponding `downgrade()`.
5. Add Pydantic schemas to `app/models/schemas.py` for the new entity.
6. Add a store or service in `app/services/`.
7. Add a router in `app/routers/`.
8. Add tests.

### 11.3 Change the embedding model

1. Deploy the new model to Triton at the path in `TRITON_TEXT_EMBEDDING_TOKENIZER_PATH`.
2. Update `TRITON_TEXT_EMBEDDING_MODEL` if the model name changed.
3. In `TritonTextEmbedder`, update `self._dim` to the new dimension.
4. Create a new Alembic migration to:
   - Alter `embedding` and `description_embedding` columns to `vector(N)` where N is the new dimension
   - Drop and recreate the DiskANN indexes
5. Update the Dockerfile to download the new tokenizer.

### 11.4 Add a new dependency

1. Add the dependency to `pyproject.toml`.
2. Run `uv lock`.
3. If it has a lazy import (optional), guard the import with a comment.

---

## 12. What NOT to do

**Architecture and layering.**
- Do not add SQLAlchemy ORM models. Use raw SQL via psycopg cursors.
- Do not add auth middleware. This is an internal LAN service; auth is handled upstream by Cognitive Companion.
- Do not add YAML config files. Use pydantic-settings with `.env`.
- Do not create circular imports between routers and services.

**Database.**
- Do not use `op.create_table()` or SQLAlchemy DDL in migrations. Use `op.execute()` with raw SQL.
- Do not run migrations by hand in production. They run on startup in the lifespan.
- Do not add foreign keys to tables outside the `semantic_memory` database.

**Error handling.**
- Do not let infrastructure exceptions bubble to the API. Catch them and raise the appropriate store error.
- Do not use bare `except:` or `except Exception: pass`. Log and return a zero value or re-raise as a typed error.

**Dependencies.**
- Do not add a runtime dependency without updating `pyproject.toml` and running `uv lock`.
- Do not import FastAPI types in services. Services operate on plain Python types and Pydantic models.

**Tests.**
- Do not add testcontainers or real database connections in tests. Mock the pool.
- Do not skip writing tests for new endpoints.

---

## 13. Where to look when stuck

| You want to ... | Read |
| --- | --- |
| Understand startup wiring | `app/main.py` (lifespan) |
| Add an endpoint | `app/routers/observations.py` (the most complete router), then `app/models/schemas.py` |
| Add a store | `app/services/observation_store.py` (the simplest store pattern) |
| Understand search query building | `app/services/search.py` |
| Understand text embedding | `app/services/text_embedder.py` |
| Understand DB connection lifecycle | `app/db/connection.py` |
| Add a migration | `app/db/alembic/versions/0001_initial_schema.py` (use as template) |
| Understand Alembic config | `app/db/alembic/env.py` |
| Fix settings shape | `app/config/config.py` |
