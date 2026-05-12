# Semantic Memory Service

Time-series searchable memory for [Cognitive Companion](https://silvermind-project.github.io). Stores scene observations, person movement transitions, and object presence state. Provides vector similarity search over image and text embeddings.

Documentation: [silvermind-project.github.io](https://silvermind-project.github.io). Agent reference: [AGENTS.md](AGENTS.md). Agent quick-start: [CLAUDE.md](CLAUDE.md).

---

## Architecture

```mermaid
flowchart LR
    subgraph Writes["Write Path"]
        CC_O["Cognitive Companion<br/>pipeline steps"] --> |POST /observations| OBS["ObservationStore"]
        CC_M["Cognitive Companion<br/>pipeline steps"] --> |POST /movements| MOV["MovementStore"]
        OBS --> |UPSERT| OP["ObjectPresenceStore"]
    end

    subgraph Reads["Read Path"]
        CC_S["Cognitive Companion<br/>pipeline + MCP"] --> |POST /observations/search| SRCH["SearchService"]
        CC_T["Cognitive Companion<br/>pipeline + MCP"] --> |GET /movements/transitions| MOV
        CC_R["Cognitive Companion<br/>pipeline + MCP"] --> |GET /objects/{room}/recent| OP
    end

    subgraph Infra["Infrastructure"]
        PG["PostgreSQL 18<br/>pgvectorscale<br/>StreamingDiskANN"]
        TRITON["Triton Inference Server<br/>embeddinggemma-300m"]
        SRCH --> |text embeddings| TRITON
        OBS --> PG
        MOV --> PG
        OP --> PG
        SRCH --> PG
    end
```

Scene observations, person movements, and object presence are written through the REST API and indexed for vector similarity search. Search supports CLIP image embeddings, text embeddings (via Triton), and metadata filters (room, time, object labels, hazard flags) with cosine distance ordering via StreamingDiskANN indexes.

---

## Prerequisites

| Component | Purpose |
| --- | --- |
| Python 3.14 | Runtime |
| PostgreSQL 18 with pgvectorscale | Application database (shared `timescale/timescaledb-ha:pg18` instance) |
| Triton Inference Server | Text embedding model (embeddinggemma-300m) |
| Docker + Docker Compose | Container runtime |
| NVIDIA GPU | Required by Triton for embedding inference |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/SilverMind-Project/cognitive-companion.git
cd cognitive-companion/semantic-memory-service

# 2. Configure
cp .env.example .env   # set DATABASE_URL, SMS_DB_USER, SMS_DB_PASSWORD

# 3. Start shared PostgreSQL (from monorepo root)
cd .. && docker compose -f docker-compose.db.yml -p nanai up -d

# 4. Start the service
cd semantic-memory-service && docker compose up -d

# 5. Verify
curl http://localhost:8400/health
```

### Local development

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8400 --reload
```

Interactive API docs at `http://localhost:8400/docs`.

---

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/v1/observations/` | Create a scene observation |
| `POST` | `/api/v1/observations/search` | Vector similarity search with metadata filters |
| `DELETE` | `/api/v1/observations/prune` | Prune observations older than N days |
| `POST` | `/api/v1/movements/` | Create a person movement record |
| `GET` | `/api/v1/movements/transitions` | Query movement transitions by person |
| `GET` | `/api/v1/objects/{room_id}/recent` | Get recent object presence in a room |
| `GET` | `/health` | Health check |

Full API reference and step-type descriptions: [silvermind-project.github.io](https://silvermind-project.github.io/features/pipeline).

---

## Configuration

| Env var | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | (required) | PostgreSQL connection string for the `semantic_memory` database |
| `TEXT_EMBEDDING_ENABLED` | `true` | Set to `false` to disable Triton text embeddings |
| `TRITON_URL` | `localhost:8701` | Triton Inference Server gRPC endpoint |
| `TRITON_TEXT_EMBEDDING_MODEL` | `embeddinggemma-300m` | Triton model name |
| `RETENTION_DAYS` | `90` | Default data retention period |

---

## Development

```bash
uv run pytest              # full test suite
uv run ruff check .        # lint
uv run mypy app/           # type check

# Database migrations
uv run alembic revision -m "description"
uv run alembic upgrade head
```

Migrations run automatically on application startup. No manual migration step is needed in deployment.

---

## License

AGPL-3.0-or-later
