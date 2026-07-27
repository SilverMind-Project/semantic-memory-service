import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.config.config import settings

config = context.config
# fileConfig() resets root's handlers/level and disables pre-existing loggers.
# The CLI wants that; in-process callers (app.main lifespan) do not, since it
# would silence the app and uvicorn loggers for the rest of the process.
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name)

target_metadata = None


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
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
