"""Database migration runner using Alembic."""

import asyncio
import logging
from pathlib import Path

from alembic.config import Config
from alembic import command


logger = logging.getLogger(__name__)

_ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def _get_alembic_config() -> Config:
    cfg = Config(str(_ALEMBIC_INI))
    # Migrations run inside the app process, so env.py must leave the already
    # configured app and uvicorn loggers alone. See env.py for the guard.
    cfg.attributes["configure_logging"] = False
    return cfg


async def run_migrations() -> None:
    """Apply all pending migrations via Alembic."""
    alembic_cfg = _get_alembic_config()
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")
