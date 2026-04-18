# Semantic Memory Service

This microservice provides long-term temporal and semantic context for the Cognitive Companion system. It uses PostgreSQL with `pgvector` to store and query high-dimensional CLIP embeddings, structured scene observations, and person movement transitions.

## Features

- **Semantic Search**: Perform vector similarity searches using CLIP embeddings.
- **Temporal Context**: Query observations and movements within specific time windows.
- **Object Tracking**: Aggregate and track the "last seen" status of objects in specific rooms.
- **Movement Inference**: Track semantic room transitions (entering/exiting) for person identification and activity enrichment.
- **Retention Management**: Automated data pruning based on configurable retention policies.
- **High-Performance**: Built with FastAPI, `asyncpg` for asynchronous database access, and `uv` for lightning-fast dependency management.

## Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Framework**: FastAPI
- **Database**: PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)
- **Linting/Formatting**: [Ruff](https://github.com/astral-sh/ruff)
- **Static Analysis**: [Mypy](https://mypy-lang.org/)
- **Testing**: Pytest
- **Containerization**: Docker

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker & Docker Compose
- PostgreSQL 15+ with `pgvector` extension enabled

### Local Development

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   The API will be available at `http://localhost:8300`.

3. **Run tests**:
   ```bash
   uv run pytest
   ```

4. **Linting and Type Checking**:
   ```bash
   uv run ruff check .
   uv run mypy .
   ```

## API Documentation

Once the service is running, you can access the interactive Swagger documentation at:
`http://localhost:8300/docs`

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

> **Note**: The `movements` router exists in `app/routers/movements.py` but is not yet registered in `app/main.py`. These endpoints are currently unavailable.

### Objects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/objects/{room_id}/recent` | Get recent object presence in a room |

> **Note**: The `objects` router exists in `app/routers/objects.py` but is not yet registered in `app/main.py`. These endpoints are currently unavailable.

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

- `201 Created`: Resource successfully created
- `400 Bad Request`: Invalid input or store operation failed
- `500 Internal Server Error`: Service-level error (e.g., search failure)

### Custom Exceptions

Each service layer has its own exception type for targeted error handling:

- `ObservationStoreError`: Observation creation/retrieval failures
- `MovementStoreError`: Movement record failures
- `SearchServiceError`: Vector search failures
- `ObjectPresenceStoreError`: Object presence tracking failures

## Architecture

The service follows a layered architecture:

1. **API Layer (Routers)**: Defines RESTful endpoints with request validation and error handling.
2. **Service Layer (Stores/Services)**: Implements business logic and database operations.
3. **Data Access Layer (Connection)**: Manages PostgreSQL connection pooling via `asyncpg`.
4. **Persistence Layer (PostgreSQL)**: Stores all structured and vector data using `pgvector`.

### Lifecycle Management

The application uses FastAPI's lifespan context manager to handle:
- Database connection initialization on startup
- Graceful shutdown with connection cleanup

### Configuration

Configuration is managed via [pydantic-settings](https://docs.pydantic.dev/dev-v2/usage/pydantic_settings/):

- `DATABASE_URL`: PostgreSQL connection string
- `API_V1_STR`: API version prefix (default: `/api/v1`)
- `PROJECT_NAME`: Service name for API documentation
- `RETENTION_DAYS`: Default data retention period
- `TEXT_EMBEDDING_MODEL`: Sentence-transformers model ID (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `TEXT_EMBEDDING_ENABLED`: Enable text embedding fallback (default: `true`)

## Database Schema

The service uses the following tables:

- `scene_observations`: Stores scene observations with CLIP embeddings
- `person_movements`: Tracks person movement transitions between rooms
- `object_presence`: Aggregates object presence by room

## Development Standards

This service adheres to high engineering standards:

- **Type Safety**: Full type annotations using Python 3.12+ syntax
- **Async/Await**: Asynchronous database operations using `asyncpg`
- **Error Handling**: Centralized exception handlers with structured logging
- **Code Quality**: Enforced via Ruff linting and Mypy type checking
- **Testing**: Comprehensive test coverage for API and service layers

## Known Issues

- `aiosqlite` is listed as a dependency in `pyproject.toml` but never used - the service uses `asyncpg` exclusively.
- Mypy is mentioned in the Development Standards but not listed in dev dependencies in `pyproject.toml`.
- The `movements` and `objects` router modules exist but are not registered in `app/main.py` - their endpoints are currently unavailable.
- The Dockerfile uses `python:3.11-slim` but `pyproject.toml` requires `>=3.12`.
