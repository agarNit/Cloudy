import uuid
from datetime import datetime, timezone

import aiosqlite

from cloudy.memory.db import db_path as _db_path
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_active_at TEXT NOT NULL
        )
        """
    )
    await db.commit()


async def new_session() -> str:
    """Create and register a brand new session. This is the default on every launch."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(_db_path()) as db:
        await _ensure_table(db)
        await db.execute(
            "INSERT INTO sessions (session_id, created_at, last_active_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        await db.commit()
    logger.info(f"Started new session: {session_id}")
    return session_id


async def switch_session(session_id: str) -> str:
    """Switch to a session id, registering it if it isn't already known."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(_db_path()) as db:
        await _ensure_table(db)
        await db.execute(
            """
            INSERT INTO sessions (session_id, created_at, last_active_at) VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET last_active_at = excluded.last_active_at
            """,
            (session_id, now, now),
        )
        await db.commit()
    logger.info(f"Switched to session: {session_id}")
    return session_id


async def touch_session(session_id: str) -> None:
    """Bump last_active_at — called whenever a session is actually used (a query is handled)."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(_db_path()) as db:
        await _ensure_table(db)
        await db.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        await db.commit()


async def list_sessions(limit: int = 20) -> list[dict]:
    """Return past sessions for this project, most recently active first."""
    async with aiosqlite.connect(_db_path()) as db:
        await _ensure_table(db)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT session_id, created_at, last_active_at FROM sessions "
            "ORDER BY last_active_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def most_recent_session() -> str | None:
    """Return the most recently active session id, or None if no sessions exist yet."""
    sessions = await list_sessions(limit=1)
    return sessions[0]["session_id"] if sessions else None
