from psycopg_pool import AsyncConnectionPool
from app.config.config import settings


class Database:
    def __init__(self):
        self.pool: AsyncConnectionPool | None = None

    async def connect(self):
        if not self.pool:
            self.pool = AsyncConnectionPool(settings.DATABASE_URL, open=False)
            await self.pool.open()

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    def get_pool(self) -> AsyncConnectionPool:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")
        return self.pool


db = Database()
