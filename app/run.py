"""Application entry point with proper lifecycle management."""

import uvicorn
from app.main import app
from app.db.connection import db


async def main() -> None:
    """Run the application with proper database lifecycle."""
    await db.connect()
    try:
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8300,
            lifespan="on",
        )
        server = uvicorn.Server(config)
        await server.serve()
    finally:
        await db.disconnect()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
