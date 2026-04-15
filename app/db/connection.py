import asyncpg
from app.config.config import settings

class Database:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(settings.DATABASE_URL)

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    def get_pool(self) -> asyncpg.Pool:
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")
        return self.pool

db = Database()
