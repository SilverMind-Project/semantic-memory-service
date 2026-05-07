"""Database migration runner using Alembic.

Handles transition from legacy ``_schema_version`` tracking on first run,
then delegates to Alembic for all subsequent migrations.
"""

import asyncio
import logging
from pathlib import Path

from alembic.config import Config
from alembic import command

from app.db.connection import db

logger = logging.getLogger(__name__)

_ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"

_LEGACY_TO_ALEMBIC = {
    "001_initial.sql": "0610b1b70adf",
    "002_description_embedding.sql": "957e5f52208f",
}


def _get_alembic_config() -> Config:
    return Config(str(_ALEMBIC_INI))


async def _has_legacy_tracking() -> bool:
    async with db.pool.connection() as conn:
        row = await conn.fetchrow(
            "SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = '_schema_version')"
        )
        return row[0]


async def _transition_from_legacy() -> None:
    """Stamp Alembic to match legacy state, then drop the old tracking table."""
    async with db.pool.connection() as conn:
        rows = await conn.fetch(
            "SELECT filename FROM _schema_version ORDER BY filename"
        )
        applied = [r["filename"] for r in rows]

    alembic_cfg = _get_alembic_config()

    if not applied:
        logger.info("Legacy _schema_version table exists but is empty, dropping it")
    else:
        last_legacy = applied[-1]
        alembic_rev = _LEGACY_TO_ALEMBIC.get(last_legacy)
        if alembic_rev is None:
            logger.warning(
                "Unknown legacy migration %s, stamping alembic at base", last_legacy
            )
            alembic_rev = "base"

        logger.info(
            "Transitioning from legacy: %s → alembic %s", last_legacy, alembic_rev
        )
        await asyncio.to_thread(command.stamp, alembic_cfg, alembic_rev)

    async with db.pool.connection() as conn:
        await conn.execute("DROP TABLE _schema_version")
    logger.info("Dropped legacy _schema_version table")


async def run_migrations() -> None:
    """Apply all pending migrations via Alembic, handling legacy transition."""
    if await _has_legacy_tracking():
        await _transition_from_legacy()

    alembic_cfg = _get_alembic_config()
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")


async def init_db() -> None:
    """Legacy entry point kept for compatibility — delegates to run_migrations."""
    await run_migrations()


if __name__ == "__main__":
    async def main():
        await db.connect()
        await run_migrations()
        await db.disconnect()

    asyncio.run(main())
