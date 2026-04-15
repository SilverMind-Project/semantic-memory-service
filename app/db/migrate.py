import os
from app.db.connection import db

async def init_db():
    # In a real app, we would use a migration tool like Alembic
    # For now, we'll just run the initial migration script
    migration_path = os.path.join(os.path.dirname(__file__), 'migrations', '001_initial.sql')
    with open(migration_path, 'r') as f:
        sql = f.read()
    
    async with db.pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(sql)
    print("Database initialized.")

if __name__ == "__main__":
    import asyncio
    from app.config.config import settings

    async def main():
        await db.connect()
        await init_db()
        await db.disconnect()

    asyncio.run(main())
