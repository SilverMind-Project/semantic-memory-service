"""FastAPI application entry point with lifecycle management and error handling."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config.config import settings
from app.db.connection import db
from app.db.migrate import run_migrations
from app.routers import observations, movements, objects, stats
from app.services.observation_store import ObservationStoreError
from app.services.movement_store import MovementStoreError
from app.services.search import SearchServiceError
from app.services.object_presence import ObjectPresenceStoreError
from app.services.stats_store import StatsStoreError

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
    await run_migrations()
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
@app.exception_handler(MovementStoreError)
@app.exception_handler(SearchServiceError)
@app.exception_handler(ObjectPresenceStoreError)
@app.exception_handler(StatsStoreError)
async def service_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("%s: %s", type(exc).__name__, exc)
    status_code = 500 if isinstance(exc, (SearchServiceError, StatsStoreError)) else 400
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


app.include_router(observations.router, prefix=settings.API_V1_STR)
app.include_router(movements.router, prefix=settings.API_V1_STR)
app.include_router(objects.router, prefix=settings.API_V1_STR)
app.include_router(stats.router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status of the service.
    """
    return {"status": "healthy", "service": settings.PROJECT_NAME}
