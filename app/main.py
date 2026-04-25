"""FastAPI application entry point with lifecycle management and error handling."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Import routers (imported at top to avoid circular dependencies)
from app.routers import observations  # noqa: E402

from app.config.config import settings
from app.db.connection import db
from app.services.observation_store import ObservationStoreError
from app.services.movement_store import MovementStoreError
from app.services.search import SearchServiceError
from app.services.object_presence import ObjectPresenceStoreError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle.

    Args:
        app: The FastAPI application.
    """
    logger.info("Starting semantic memory service")
    await db.connect()
    logger.info("Database connection established")
    yield
    logger.info("Shutting down semantic memory service")
    await db.disconnect()
    logger.info("Database connection closed")


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
)


@app.exception_handler(ObservationStoreError)
async def observation_store_exception_handler(
    request: Request,
    exc: ObservationStoreError,
) -> JSONResponse:
    """Handle observation store errors."""
    logger.error(f"Observation store error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(MovementStoreError)
async def movement_store_exception_handler(
    request: Request,
    exc: MovementStoreError,
) -> JSONResponse:
    """Handle movement store errors."""
    logger.error(f"Movement store error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(SearchServiceError)
async def search_service_exception_handler(
    request: Request,
    exc: SearchServiceError,
) -> JSONResponse:
    """Handle search service errors."""
    logger.error(f"Search service error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.exception_handler(ObjectPresenceStoreError)
async def object_presence_store_exception_handler(
    request: Request,
    exc: ObjectPresenceStoreError,
) -> JSONResponse:
    """Handle object presence store errors."""
    logger.error(f"Object presence store error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


# Include routers in app
app.include_router(observations.router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status of the service.
    """
    return {"status": "healthy", "service": settings.PROJECT_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8400)
