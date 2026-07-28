import asyncio
from logging.config import fileConfig

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from alembic.ddl.impl import DefaultImpl
from app.config.config import settings

config = context.config
# fileConfig() resets root's handlers/level and disables pre-existing loggers.
# The CLI wants that; in-process callers (app.main lifespan) do not, since it
# would silence the app and uvicorn loggers for the rest of the process.
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name)

target_metadata = None

# -- Revision id length -------------------------------------------------------
# Alembic hardcodes ``Column("version_num", String(32))`` in
# ``alembic.ddl.impl.DefaultImpl.version_table_impl``. A revision id longer than
# 32 characters passes file generation and graph resolution, then fails at apply
# time when Alembic writes the head, and transactional DDL rolls the whole
# migration back. ``version_table_impl`` is a documented override hook (added in
# Alembic 1.14), so widen the column there.
VERSION_NUM_LENGTH = 128

_original_version_table_impl = DefaultImpl.version_table_impl


def _wide_version_table_impl(self, **kw):
    """Return alembic's version table with a wider ``version_num`` column."""
    table = _original_version_table_impl(self, **kw)
    table.c.version_num.type = sa.String(VERSION_NUM_LENGTH)
    return table


DefaultImpl.version_table_impl = _wide_version_table_impl


def _widen_existing_version_table(connection) -> None:
    """Widen ``version_num`` on databases created before the override landed.

    The override only affects table *creation*; a database whose
    ``alembic_version`` already exists keeps ``varchar(32)`` until altered.
    No-op when the table is absent (fresh database) or already wide enough.

    Online mode only. ``alembic upgrade --sql`` emits SQL without connecting, so
    an offline script does not carry this ALTER; widen such databases by hand.
    """
    current_length = connection.exec_driver_sql(
        "SELECT character_maximum_length FROM information_schema.columns "
        "WHERE table_name = 'alembic_version' AND column_name = 'version_num'"
    ).scalar()
    if current_length is None or current_length >= VERSION_NUM_LENGTH:
        return
    connection.exec_driver_sql(
        f"ALTER TABLE alembic_version ALTER COLUMN version_num "
        f"TYPE VARCHAR({VERSION_NUM_LENGTH})"
    )


def _get_async_url() -> str:
    url = settings.DATABASE_URL
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    elif url.startswith("postgresql+psycopg_async://"):
        pass
    else:
        url = url.replace("postgresql://", "postgresql+psycopg_async://")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_get_async_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = create_async_engine(_get_async_url())
    # Widen on a separate connection that we commit ourselves. Issuing any
    # statement on the migration connection before context.configure() starts a
    # transaction Alembic does not own, and it then silently declines to commit
    # the migration.
    async with connectable.connect() as connection:
        await connection.run_sync(_widen_existing_version_table)
        await connection.commit()

    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
