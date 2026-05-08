"""Database migration runner using Alembic."""

import asyncio
import logging
from pathlib import Path

from alembic.config import Config
from alembic import command


logger = logging.getLogger(__name__)

_ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


def _get_alembic_config() -> Config:
    return Config(str(_ALEMBIC_INI))


async def run_migrations() -> None:
    """Apply all pending migrations via Alembic."""
    alembic_cfg = _get_alembic_config()
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")
