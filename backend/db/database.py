"""
Database module — manages the PostgreSQL connection pool for LangGraph checkpointing.
Uses psycopg3 (psycopg) which is required by langgraph-checkpoint-postgres.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import psycopg_pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config import DATABASE_URL

logger = logging.getLogger(__name__)

# Global async connection pool (shared across all WebSocket sessions)
_pool: psycopg_pool.AsyncConnectionPool | None = None


async def get_pool() -> psycopg_pool.AsyncConnectionPool:
    """Return the global async connection pool, initializing it if needed."""
    global _pool
    if _pool is None or _pool.closed:
        _pool = psycopg_pool.AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=2,
            max_size=20,
            # Reconnect if connection goes idle > 5 min
            reconnect_timeout=30,
            open=False,  # We call open() explicitly
        )
        await _pool.open()
        logger.info("✅ PostgreSQL connection pool initialized (min=2, max=20)")
    return _pool


async def close_pool():
    """Gracefully close the connection pool on shutdown."""
    global _pool
    if _pool and not _pool.closed:
        await _pool.close()
        logger.info("🔒 PostgreSQL connection pool closed")


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """
    Context manager that yields a LangGraph AsyncPostgresSaver.
    Use this inside each WebSocket session to get an isolated checkpointer.
    """
    pool = await get_pool()
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
        # Create tables on first use (idempotent)
        await saver.setup()
        yield saver


async def init_db():
    """
    Run once on startup to ensure all LangGraph checkpoint tables exist.
    LangGraph needs: checkpoints, checkpoint_writes, checkpoint_blobs
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        # Enable pgvector extension for future RAG migration
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        logger.info("✅ Database extensions verified (vector, pg_trgm)")

    # Initialize LangGraph checkpoint schema
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
        await saver.setup()
        logger.info("✅ LangGraph checkpoint tables initialized")
