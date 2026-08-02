import asyncio
import json
from datetime import datetime, timezone

import aiosqlite

from cloudy.context.freshness import get_index_generation
from cloudy.llm.factory import get_embedder, EMBED_LOCK
from cloudy.memory.db import db_path
from cloudy.memory.semantic import get_memory_generation
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

# Only these tools may appear in a turn for it to be cache-eligible. Deliberately
# an allowlist, not a denylist — the cost of under-caching is a missed speedup
# (safe); the cost of over-caching is a confidently wrong answer served for
# something that actually depends on untracked external state (a shell command's
# output, a live GitHub issue list). When in doubt, this must say no.
#
# Includes the MCP filesystem server's own read tools (read_text_file, etc.) —
# confirmed via a real run that the model reaches for these over the local
# read_file/list_directory tools often enough to matter. These are safe to
# include because they read the same local files freshness already tracks, so
# the index_generation fingerprint covers them. Deliberately does NOT include
# any MCP GitHub read tool (list_issues, get_pull_request, ...) — those reflect
# live external state with no fingerprint tracking it at all, so caching them
# would go stale the moment something changes on GitHub's side, silently.
CACHEABLE_TOOL_NAMES = frozenset({
    "search_codebase", "read_file", "list_directory", "file_exists",
    "find_session", "recall_memory", "load_skill",
    # MCP filesystem server — read-only, same local files freshness tracks
    "read_text_file", "read_media_file", "read_multiple_files",
    "list_directory_with_sizes", "directory_tree", "search_files",
    "get_file_info", "list_allowed_directories",
})

# Conservative on purpose — a false-positive "hit" (two similar-sounding
# questions that actually want different answers) is worse than a missed hit.
# Calibrated empirically against all-MiniLM-L6-v2, not assumed: real paraphrase
# pairs scored 0.91-0.95, genuinely different questions on the same topic
# scored ~0.50, and unrelated questions scored ~0.04 — 0.85 catches the former
# with a comfortable margin above the latter. This will still miss more
# loosely-worded paraphrases (e.g. "run the tests" vs "execute the test
# suite" scored only 0.70) — an accepted tradeoff given the priority is no
# false positives, not maximum hit rate.
SIMILARITY_THRESHOLD = 0.85


def is_cacheable_turn(tool_names: set[str]) -> bool:
    return tool_names.issubset(CACHEABLE_TOOL_NAMES)


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            question_embedding TEXT NOT NULL,
            answer TEXT NOT NULL,
            index_generation INTEGER NOT NULL,
            memory_generation INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    await db.commit()


def _embed(embedder, text: str) -> list[float]:
    with EMBED_LOCK:
        return embedder.embed_query(text)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def lookup(question: str) -> str | None:
    """Return a cached answer for a near-duplicate question, but only if
    nothing tracked (index state, long-term memory) has changed since it was
    cached — checked via generation counters, not by re-verifying content.
    """
    index_gen = await get_index_generation()
    memory_gen = await get_memory_generation()

    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT question, question_embedding, answer FROM semantic_cache "
            "WHERE index_generation = ? AND memory_generation = ?",
            (index_gen, memory_gen),
        )
        rows = await cursor.fetchall()

    if not rows:
        return None

    embedder = get_embedder()
    query_vec = await asyncio.to_thread(_embed, embedder, question)

    best_score, best_answer = 0.0, None
    for row in rows:
        score = _cosine(query_vec, json.loads(row["question_embedding"]))
        if score > best_score:
            best_score, best_answer = score, row["answer"]

    if best_score >= SIMILARITY_THRESHOLD:
        logger.info(f"Semantic cache hit (score={best_score:.3f}): {question}")
        return best_answer
    return None


async def store(question: str, answer: str) -> None:
    index_gen = await get_index_generation()
    memory_gen = await get_memory_generation()

    embedder = get_embedder()
    vec = await asyncio.to_thread(_embed, embedder, question)
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        await db.execute(
            """
            INSERT INTO semantic_cache
                (question, question_embedding, answer, index_generation, memory_generation, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (question, json.dumps(vec), answer, index_gen, memory_gen, now),
        )
        await db.commit()
    logger.info(f"Cached answer for: {question}")
