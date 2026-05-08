# Semantic Memory Service

This microservice provides long-term temporal and semantic context for the Cognitive Companion system. It uses PostgreSQL with `pgvectorscale` (StreamingDiskANN) for high-performance vector similarity search over CLIP embeddings, structured scene observations, and person movement transitions.

## Features

- **Semantic Search**: Vector similarity searches using CLIP and text embeddings (via Triton Inference Server)
- **Temporal Context**: Query observations and movements within specific time windows
- **Object Tracking**: Aggregate and track the "last seen" status of objects in specific rooms
- **Movement Inference**: Track semantic room transitions (entering/exiting) for person identification and activity enrichment
- **Retention Management**: Automated data pruning based on configurable retention policies
- **High-Performance**: Built with FastAPI, `asyncpg` for asynchronous database access, and `uv` for dependency management

## Tech Stack

- **Language**: Python 3.14+
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Framework**: FastAPI
- **Database**: PostgreSQL + [pgvectorscale](https://github.com/timescale/pgvectorscale) (StreamingDiskANN indexes)
- **Migrations**: [Alembic](https://alembic.sqlalchemy.org/) with async psycopg3
- **Linting/Formatting**: [Ruff](https://github.com/astral-sh/ruff)
- **Static Analysis**: [Mypy](https://mypy-lang.org/)
- **Testing**: Pytest + pytest-asyncio
- **Containerization**: Docker

## Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv)
- Docker & Docker Compose
- PostgreSQL with pgvectorscale extension

### Local Development

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run with Docker Compose**:
   ```bash
   docker compose up --build
   ```
   The API will be available at `http://localhost:8400`.

3. **Run tests**:
   ```bash
   uv run pytest
   ```

4. **Linting and type checking**:
   ```bash
   uv run ruff check .
   uv run mypy app/ --ignore-missing-imports --explicit-package-bases
   ```

## API Documentation

Once the service is running, interactive Swagger documentation is available at:
`http://localhost:8400/docs`

## API Endpoints

### Observations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/observations/` | Create a new scene observation |
| POST | `/api/v1/observations/search` | Search observations using vector similarity |
| DELETE | `/api/v1/observations/prune` | Prune observations older than N days |

### Movements

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/movements/` | Create a new movement record |
| GET | `/api/v1/movements/transitions` | Get movement transitions for a person |

### Objects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/objects/{room_id}/recent` | Get recent object presence in a room |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check endpoint |

## Error Handling

The service uses standard HTTP status codes and returns errors in a consistent format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource successfully created
- `400 Bad Request`: Invalid input or store operation failed
- `422 Unprocessable Entity`: Request validation error
- `500 Internal Server Error`: Service-level error (e.g., search failure)

## Architecture

The service follows a layered architecture:

1. **API Layer (Routers)**: RESTful endpoints with request validation and error handling
2. **Service Layer (Stores/Services)**: Business logic and database operations
3. **Data Access Layer (Connection)**: PostgreSQL connection pooling via psycopg3
4. **Persistence Layer (PostgreSQL)**: Structured and vector data using pgvectorscale

### Lifecycle Management

The application uses FastAPI's lifespan context manager to handle:
- Database connection initialization on startup
- Automatic Alembic migration on startup
- Graceful shutdown with connection cleanup

### Configuration

Configuration is managed via [pydantic-settings](https://docs.pydantic.dev/dev-v2/usage/pydantic_settings/):

- `DATABASE_URL`: PostgreSQL connection string (default: `postgresql://postgres:postgres@localhost:5432/semantic_memory`)
- `API_V1_STR`: API version prefix (default: `/api/v1`)
- `PROJECT_NAME`: Service name for API documentation (default: `semantic-memory-service`)
- `RETENTION_DAYS`: Default data retention period (default: `90`)
- `TEXT_EMBEDDING_ENABLED`: Enable text embedding fallback (default: `true`)
- `TRITON_URL`: Triton Inference Server gRPC endpoint (default: `localhost:8701`)
- `TRITON_TEXT_EMBEDDING_MODEL`: Triton model name for text embeddings (default: `embeddinggemma-300m`)
- `TRITON_TEXT_EMBEDDING_TOKENIZER_PATH`: Path to the tokenizer.json for the embedding model

## Database Migrations

Database schema changes are managed with Alembic. Migrations run automatically on application startup.

### Creating a new migration

```bash
uv run alembic revision -m "description of the change"
```

This generates a new file in `app/db/alembic/versions/`. Fill in the `upgrade()` and `downgrade()` methods.

To apply migrations manually:

```bash
uv run alembic upgrade head
```

### Schema

- `scene_observations`: Scene observations with CLIP and text embeddings
- `person_movements`: Person movement transitions between rooms
- `object_presence`: Object presence aggregated by room
- `person_current_location`: Materialized view for current person locations
