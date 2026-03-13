"""
Edge Crew — Postgres connection pool and helpers.
Uses asyncpg for async connection pooling.
Gracefully degrades if DATABASE_URL is not set (all operations return None/[]).
"""
import os
import asyncpg
import logging

logger = logging.getLogger("edge-crew")

_pool = None


async def get_pool():
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            logger.warning("DATABASE_URL not set — DB features disabled")
            return None
        # Render uses postgres:// but asyncpg needs postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        _pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Postgres pool created")
    return _pool


async def close_pool():
    """Close the connection pool on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Postgres pool closed")


async def execute(query: str, *args):
    """Execute a query (INSERT/UPDATE/DELETE)."""
    pool = await get_pool()
    if not pool:
        return None
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args):
    """Fetch multiple rows."""
    pool = await get_pool()
    if not pool:
        return []
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args):
    """Fetch a single row."""
    pool = await get_pool()
    if not pool:
        return None
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetchval(query: str, *args):
    """Fetch a single value."""
    pool = await get_pool()
    if not pool:
        return None
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def executemany(query: str, args_list):
    """Execute a query for multiple rows."""
    pool = await get_pool()
    if not pool:
        return None
    async with pool.acquire() as conn:
        return await conn.executemany(query, args_list)


async def init_schema():
    """Run schema.sql to create tables if they don't exist."""
    pool = await get_pool()
    if not pool:
        logger.warning("No DB pool — skipping schema init")
        return
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        logger.error(f"schema.sql not found at {schema_path}")
        return
    with open(schema_path, "r") as f:
        sql = f.read()
    async with pool.acquire() as conn:
        await conn.execute(sql)
    logger.info("Schema initialized")
