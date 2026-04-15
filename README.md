# Semantic Memory Service

This microservice provides long-term temporal and semantic context for the Cognitive Companion system. It uses PostgreSQL with `pgvector` to store and query high-dimensional CLIP embeddings, structured scene observations, and person movement transitions.

## Features

- **Semantic Search**: Perform vector similarity searches using CLIP embeddings.
- **Temporal Context**: Query observations and movements within specific time windows.
- **Object Tracking**: Aggregate and track the "last seen" status of objects in specific rooms.
- **Movement Inference**: Track semantic room transitions (entering/exiting) for person identification and activity enrichment.
- **High-Performance**: Built with FastAPI, `asyncpg` for asynchronous database access, and `uv` for lightning-fast dependency management.

## Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Framework**: FastAPI
- **Database**: PostgreSQL + [pgvector](httpsran/pgvector)
- **Linting/Formatting**: [Ruff](https://github.com/astral-sh/ruff)
- **Static Analysis**: [Mypy](https://mypy-lang.org/)
- **Testing**: Pytest

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker & Docker Compose

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

## Architecture

The service follows a layered architecture:
1. **API Layer (Routers)**: Defines RESTful endpoints.
2. **Service Layer (Stores/Services)**: Implements business logic and complex queries (Search, Movement Tracking).
3. **Data Access Layer (Connection)**: Manages PostgreSQL connection pooling.
4. **Persistence Layer (PostgreSQL)**: Stores all structured and vector data.
