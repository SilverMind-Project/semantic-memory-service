from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.connection import db
from app.routers import observations, movements, objects
from app.config.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB connection pool
    await db.connect()
    yield
    # Shutdown: Close DB connection pool
    await db.disconnect()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    lifespan=lifespan
)

# Register Routers
app.include_router(observations.router, prefix=settings.API_V1_STR)
app.include_router(movements.router, prefix=settings.API_V1_STR)
app.include_router(objects.router, prefix=settings.API_V1_STR)

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}

if __name__t:
    import uvicorn
    import sys
    uvicorn.run(app, host="0.0.0.0", port=8300)
