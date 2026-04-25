import os
from app.db.connection import db


async def init_db():
    migration_path = os.path.join(os.path.dirname(__file__), 'migrations', '001_initial.sql')
    with open(migration_path, 'r') as f:
        sql = f.read()

    async with db.pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(sql)
    print("Database initialized.")


if __name__ == "__main__":
    import asyncio

    async def main():
        await db.connect()
        await init_db()
        await db.disconnect()

    asyncio.run(main())
