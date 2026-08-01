import asyncio
import json
from datetime import datetime, timezone

import aiosqlite

from cloudy.memory.db import db_path as _db_path
from cloudy.llm.factory import get_llm, get_embedder
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


_SUMMARY_PROMPT = (
    "Summarize what was worked on or decided in this conversation, in 1-2 plain "
    "sentences. Focus on the substance (what was asked, built, fixed, or decided) "
    "— not conversational filler.\n\nConversation:\n{transcript}"
)

# Defensive cap on how much transcript text goes into the summarization call —
# long sessions are already compacted by SummarizationMiddleware before this runs.
_MAX_TRANSCRIPT_CHARS = 12000


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS session_log (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            embedding TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL
        )
        """
    )
    await db.commit()


async def _get_session_messages(checkpointer, session_id: str) -> list:
    cfg = {"configurable": {"thread_id": session_id}}
    checkpoint_tuple = await checkpointer.aget_tuple(cfg)
    if checkpoint_tuple is None:
        return []
    return checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])


def _messages_to_text(messages: list) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "type", "unknown")
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        content = str(content).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _summarize(messages: list) -> str:
    transcript = _messages_to_text(messages)
    if not transcript:
        return ""
    llm = get_llm()
    prompt = _SUMMARY_PROMPT.format(transcript=transcript[:_MAX_TRANSCRIPT_CHARS])
    response = await llm.ainvoke(prompt)
    return str(response.content).strip()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def close_session(checkpointer, session_id: str) -> None:
    """Summarize and store the given session if it has any messages.

    Safe to call more than once for the same session — the row is replaced
    (upserted), not duplicated, so re-closing a resumed session just refreshes
    its summary. Never raises — this runs from shutdown paths (including
    Ctrl+C), and a summarization failure there should never block exit.
    """
    try:
        messages = await _get_session_messages(checkpointer, session_id)
        if not messages:
            return
        summary = await _summarize(messages)
        if not summary:
            return

        embedder = get_embedder()
        vector = await asyncio.to_thread(embedder.embed_query, summary)
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(_db_path()) as db:
            await _ensure_table(db)
            cursor = await db.execute(
                "SELECT started_at FROM session_log WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            started_at = row[0] if row else now
            await db.execute(
                """
                INSERT INTO session_log
                    (session_id, summary, embedding, message_count, started_at, ended_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    embedding = excluded.embedding,
                    message_count = excluded.message_count,
                    ended_at = excluded.ended_at
                """,
                (session_id, summary, json.dumps(vector), len(messages), started_at, now),
            )
            await db.commit()
        logger.info(f"Logged session {session_id}: {summary}")
    except Exception as e:
        logger.error(f"Failed to log session {session_id}: {e}")


async def backfill_missing_summaries(checkpointer) -> int:
    """Summarize any known session that has no session_log entry yet.

    Covers sessions that ended without going through close_session at all —
    terminal force-closed, process killed, crash. Run once at startup so
    summaries stay eventually-consistent regardless of how a session ended.
    """
    from cloudy.memory.session import list_sessions

    sessions = await list_sessions(limit=1000)
    async with aiosqlite.connect(_db_path()) as db:
        await _ensure_table(db)
        cursor = await db.execute("SELECT session_id FROM session_log")
        logged = {row[0] for row in await cursor.fetchall()}

    missing = [s["session_id"] for s in sessions if s["session_id"] not in logged]
    for session_id in missing:
        await close_session(checkpointer, session_id)
    if missing:
        logger.info(f"Backfilled {len(missing)} session summaries")
    return len(missing)


async def get_summaries() -> dict[str, str]:
    """session_id -> summary, for every logged session. Used to enrich /sessions."""
    async with aiosqlite.connect(_db_path()) as db:
        await _ensure_table(db)
        cursor = await db.execute("SELECT session_id, summary FROM session_log")
        rows = await cursor.fetchall()
    return {row[0]: row[1] for row in rows}


async def find_sessions(query: str, k: int = 5) -> list[dict]:
    """Semantic search over past session summaries — brute-force cosine similarity.

    Fine at this scale: a single user's session history tops out at a few
    hundred entries, so ranking them in Python is effectively instant and
    doesn't warrant standing up a vector store for it.
    """
    async with aiosqlite.connect(_db_path()) as db:
        await _ensure_table(db)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT session_id, summary, embedding, started_at, ended_at FROM session_log"
        )
        rows = await cursor.fetchall()
    if not rows:
        return []

    embedder = get_embedder()
    query_vec = await asyncio.to_thread(embedder.embed_query, query)

    scored = []
    for row in rows:
        vec = json.loads(row["embedding"])
        scored.append(
            {
                "session_id": row["session_id"],
                "summary": row["summary"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "score": _cosine(query_vec, vec),
            }
        )
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:k]
