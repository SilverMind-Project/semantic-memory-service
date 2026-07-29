# CLAUDE.md

Quick reference for Claude Code agents in `semantic-memory-service/`. The full reference is [AGENTS.md](AGENTS.md); this file is the orientation pointer plus invariants you must hold from the first edit.

---

## What this is

Time-series searchable memory for Cognitive Companion. Stores scene observations, person movement transitions, and object presence state. Provides vector similarity search over CLIP image embeddings and text embeddings, with metadata-aware filters. Designed as a write-through, query-heavy microservice serving the Cognitive Companion pipeline and MCP tools.

---

## Read before editing

1. [AGENTS.md](AGENTS.md): canonical reference (architecture, API contracts, DB schema, testing conventions, common tasks).
2. `app/main.py` lifespan: source of truth for DB connection lifecycle, migration runner, and router registration.
3. `app/services/text_embedder.py`: the `TextEmbedder` ABC, `TritonTextEmbedder` (lazy gRPC), `NullTextEmbedder` (graceful degradation), and `build_text_embedder()` factory.
4. `app/db/alembic/versions/0001_initial_schema.py`: the single migration with the full current schema.

---

## Commands

```bash
# Development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8400 --reload

# Docker
docker compose up --build               # API at http://localhost:8400

# Tests
uv run pytest                           # full suite
uv run pytest tests/test_services.py -v # services only
uv run pytest tests/test_api.py -v      # API only

# Lint / type-check
uv run ruff check .                     # lint
uv run ruff check --fix .               # fix lint
uv run mypy app/                        # settings live in [tool.mypy]

# Database
uv run alembic revision -m "description"  # autogenerate new migration
uv run alembic upgrade head               # apply pending migrations
```

---

## Non-negotiable invariants

- **Migrations always raw SQL.** Alembic migrations use `op.execute("CREATE TABLE ...")` with raw SQL, not SQLAlchemy `create_table()`. This keeps vector index definitions and pgvectorscale extensions explicit.
- **No ORM models.** The service uses raw SQL via `psycopg` async cursors. Pydantic models (`app/models/schemas.py`) are the wire format. There are no SQLAlchemy declarative models.
- **Text embedder follows ABC pattern.** `TextEmbedder` defines the interface. `TritonTextEmbedder` wraps triton-shared with lazy initialization. `NullTextEmbedder` is the disabled fallback. New embedders must implement the ABC.
- **Graceful degradation over exceptions.** Integration clients (Triton) use lazy init and return zero values (`[]`) when unavailable. No exceptions bubble from infrastructure code.
- **Migrations run on startup.** The lifespan calls `run_migrations()` which runs `alembic upgrade head` synchronously in a thread. No manual migration step needed in deployment.
- **Revision ids may be up to 128 characters.** Alembic hardcodes `Column("version_num", String(32))` in `alembic.ddl.impl.DefaultImpl.version_table_impl`. `app/db/alembic/env.py` overrides that hook so `alembic_version.version_num` is `varchar(128)`, and `_widen_existing_version_table()` runs an idempotent `ALTER` so older databases catch up. Do not run statements on the migration connection before `context.configure()`: that opens a transaction Alembic does not own, and it then reports success while committing nothing.
- **Connection pool is a singleton.** `app.db.connection.db` is the module-level `Database` instance. It owns one `AsyncConnectionPool`. Services call `db.get_pool()` and manage their own `pool.connection()` contexts.
- **Timestamp-aware storage.** `TIMESTAMPTZ` columns. Observed timestamps come from the caller (Cognitive Companion). Server-generated timestamps use `NOW()` at the SQL level.
- **Vector dimension is 768.** Both `embedding` (CLIP ViT-L/14) and `description_embedding` (embeddinggemma-300m) are 768-dimensional vectors. Changing this requires a migration and index rebuild.
- **Shared PostgreSQL instance.** The database host, port, user, and password come from `DATABASE_URL` env var. The database name is `semantic_memory` on the shared `timescale/timescaledb-ha:pg18` instance.
- **No API auth.** This is an internal microservice on the LAN. Authentication and authorization are handled by Cognitive Companion (the BFF). Do not add auth middleware.

---

## External services

| Service | Env var | Required |
| --- | --- | --- |
| PostgreSQL (shared) | `DATABASE_URL` | Required. Must have pgvectorscale extension. |
| Triton Inference Server | `TRITON_URL` (default `localhost:8701`) | Optional. Disable with `TEXT_EMBEDDING_ENABLED=false`. |

---

## Where to look when stuck

| Goal | File |
| --- | --- |
| Startup and wiring | `app/main.py` (lifespan) |
| Config shape | `app/config/config.py` (Settings) |
| DB connection pool | `app/db/connection.py` (Database) |
| Migration runner | `app/db/migrate.py` |
| Alembic env | `app/db/alembic/env.py` |
| Schema (current) | `app/db/alembic/versions/0001_initial_schema.py` |
| Wire models | `app/models/schemas.py` |
| Text embedding | `app/services/text_embedder.py` |
| Observation write | `app/services/observation_store.py` |
| Movement write/read | `app/services/movement_store.py` |
| Search with vectors | `app/services/search.py` |
| Object presence tracking | `app/services/object_presence.py` |
| Observation router | `app/routers/observations.py` |
| Movement router | `app/routers/movements.py` |
| Object router | `app/routers/objects.py` |
