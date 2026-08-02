from datetime import datetime, timezone

import aiosqlite

from cloudy.memory.db import db_path
from cloudy.llm.factory import get_llm
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

_SLUG_PROMPT = (
    "Given this new user preference and the preferences already saved below, decide "
    "whether it's the same underlying preference as one already saved.\n\n"
    "Already-saved preference slugs:\n{existing}\n\n"
    "New preference: {text}\n\n"
    "Respond with exactly two lines and nothing else:\n"
    "line 1: the EXISTING slug if this is the same preference as one already saved, "
    "otherwise a NEW short kebab-case slug (2-5 words)\n"
    "line 2: a one-sentence summary of the preference"
)

# The only three buckets a memory can go in. Kept deliberately narrow — if
# something doesn't fit one of these, it's session detail, not memory.
CATEGORY_TYPES = {"preference", "feedback", "project"}

# Bounds how much of the store gets injected into every system prompt. Facts
# get upserted by (category_type, category), so this shouldn't grow unbounded
# the way session count does — but the cap is here as a hard backstop either way.
_INDEX_LIMIT = 30


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS long_term_memory (
            category_type TEXT NOT NULL,
            category TEXT NOT NULL,
            summary TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (category_type, category)
        )
        """
    )
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_generation (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            generation INTEGER NOT NULL
        )
        """
    )
    await db.commit()


async def _bump_generation(db: aiosqlite.Connection) -> None:
    # Bumped on every save/update — semantic caching's other invalidation
    # fingerprint, alongside freshness's index_generation.
    await db.execute(
        """
        INSERT INTO memory_generation (id, generation) VALUES (1, 1)
        ON CONFLICT(id) DO UPDATE SET generation = generation + 1
        """
    )


async def get_memory_generation() -> int:
    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        cursor = await db.execute("SELECT generation FROM memory_generation WHERE id = 1")
        row = await cursor.fetchone()
    return row[0] if row else 0


async def save_memory(category_type: str, category: str, summary: str, content: str) -> str:
    """Upsert a memory entry by (category_type, category) — saving again under
    the same key updates it in place rather than creating a duplicate.
    """
    category_type = category_type.strip().lower()
    if category_type not in CATEGORY_TYPES:
        return f"Error: category_type must be one of {sorted(CATEGORY_TYPES)}, got '{category_type}'"

    category = category.strip().lower().replace(" ", "-")
    summary = summary.strip()
    content = content.strip()
    if not category or not summary or not content:
        return "Error: category, summary, and content must all be non-empty"

    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT created_at FROM long_term_memory WHERE category_type = ? AND category = ?",
            (category_type, category),
        )
        existing = await cursor.fetchone()
        created_at = existing[0] if existing else now
        await db.execute(
            """
            INSERT INTO long_term_memory
                (category_type, category, summary, content, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(category_type, category) DO UPDATE SET
                summary = excluded.summary,
                content = excluded.content,
                updated_at = excluded.updated_at
            """,
            (category_type, category, summary, content, created_at, now),
        )
        await _bump_generation(db)
        await db.commit()

    verb = "Updated" if existing else "Saved"
    logger.info(f"{verb} memory [{category_type}/{category}]: {summary}")
    return f"{verb} [{category_type}/{category}]: {summary}"


async def recall_memory(category_type: str, category: str) -> str:
    category_type = category_type.strip().lower()
    category = category.strip().lower()
    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT content FROM long_term_memory WHERE category_type = ? AND category = ?",
            (category_type, category),
        )
        row = await cursor.fetchone()
    if not row:
        return f"No memory found for [{category_type}/{category}]"
    return row[0]


async def list_memory_summaries(limit: int = _INDEX_LIMIT) -> list[dict]:
    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT category_type, category, summary FROM long_term_memory "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def remember(text: str) -> str:
    """Deterministically save free text as a preference — backs the /remember
    REPL command. There's no LLM judgment here about *whether* to save (the
    user already decided that by typing the command); the one LLM call is only
    to produce a stable dedup slug and a short summary from free-form text.
    """
    text = text.strip()
    if not text:
        return "Error: nothing to remember"

    existing = [e for e in await list_memory_summaries() if e["category_type"] == "preference"]
    existing_list = "\n".join(f"- {e['category']}: {e['summary']}" for e in existing) or "(none yet)"

    llm = get_llm()
    response = await llm.ainvoke(_SLUG_PROMPT.format(existing=existing_list, text=text))
    lines = [line.strip() for line in str(response.content).strip().splitlines() if line.strip()]
    slug = lines[0].lower().replace(" ", "-") if lines else "general-preference"
    summary = lines[1] if len(lines) > 1 else text

    return await save_memory("preference", slug, summary, text)


async def build_memory_prompt() -> str:
    """Compact, always-on index injected into the system prompt — summaries
    only, same progressive-disclosure shape as build_skills_prompt(). Full
    detail is loaded on demand via the recall_memory tool.
    """
    entries = await list_memory_summaries()
    if not entries:
        return ""
    lines = ["=== Known Project Memory ==="]
    for e in entries:
        lines.append(f"- [{e['category_type']}/{e['category']}] {e['summary']}")
    lines.append(
        "\nCall recall_memory(category_type, category) for the full detail "
        "behind any of these, if relevant."
    )
    return "\n".join(lines)
