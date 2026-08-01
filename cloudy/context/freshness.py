import hashlib
import os
import time
from pathlib import Path

import aiosqlite
from langchain_core.documents import Document
from qdrant_client.http import models

from cloudy.config import config
from cloudy.context.indexers.code_parser import parse_file, get_source_files
from cloudy.llm.factory import EMBED_LOCK
from cloudy.memory.db import db_path
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

# How long a file must sit unchanged before its new content is treated as
# "settled" enough to index. There's no commit to use as an explicit signal
# here, so this is an approximation of one — long enough that a file mid-edit
# usually isn't caught mid-thought, short enough to still feel prompt.
SETTLE_SECONDS = 8

_PAYLOAD_SOURCE_FIELD = "metadata.source"


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_files (
            file_path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            last_indexed_at TEXT NOT NULL
        )
        """
    )
    await db.commit()


def _hash_file(filepath: str) -> str:
    return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()


async def _get_manifest() -> dict[str, dict]:
    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT file_path, content_hash, mtime FROM indexed_files")
        rows = await cursor.fetchall()
    return {row["file_path"]: {"hash": row["content_hash"], "mtime": row["mtime"]} for row in rows}


async def _upsert_manifest(file_path: str, content_hash: str, mtime: float) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        await db.execute(
            """
            INSERT INTO indexed_files (file_path, content_hash, mtime, last_indexed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                content_hash = excluded.content_hash,
                mtime = excluded.mtime,
                last_indexed_at = excluded.last_indexed_at
            """,
            (file_path, content_hash, mtime, now),
        )
        await db.commit()


async def _remove_from_manifest(file_path: str) -> None:
    async with aiosqlite.connect(db_path()) as db:
        await _ensure_table(db)
        await db.execute("DELETE FROM indexed_files WHERE file_path = ?", (file_path,))
        await db.commit()


def _ensure_payload_index(client, collection_name: str) -> None:
    # Qdrant requires an explicit index on a payload field before it can be
    # filtered on — safe to call repeatedly, it's a no-op if already present.
    client.create_payload_index(
        collection_name=collection_name,
        field_name=_PAYLOAD_SOURCE_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _delete_file_chunks(client, collection_name: str, file_path: str) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key=_PAYLOAD_SOURCE_FIELD, match=models.MatchValue(value=file_path))]
            )
        ),
    )


def _index_file(vector_store, filepath: str) -> int:
    """Parse and embed a single file, returning how many chunks were added."""
    try:
        chunks = parse_file(filepath)
    except (SyntaxError, ValueError) as e:
        logger.error(f"Skipping {filepath}: {e}")
        return 0

    docs = [
        Document(
            page_content=chunk.content,
            metadata={
                "source": chunk.source,
                "name": chunk.name,
                "type": chunk.type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            },
        )
        for chunk in chunks
    ]
    if docs:
        with EMBED_LOCK:
            vector_store.add_documents(docs)
    return len(docs)


async def seed_manifest_if_empty(repo_path: str) -> None:
    """The first time freshness runs against a repo, there's no manifest yet —
    but the vector store may already be fully indexed (a fresh index_codebase
    call just built it, or it existed from before this feature shipped).
    Adopt the current on-disk state as the baseline without touching Qdrant,
    so the next real sync only reacts to genuine changes from here on.
    """
    manifest = await _get_manifest()
    if manifest:
        return
    for filepath in get_source_files(repo_path):
        try:
            stat = os.stat(filepath)
            content_hash = _hash_file(filepath)
        except (FileNotFoundError, OSError):
            continue
        await _upsert_manifest(filepath, content_hash, stat.st_mtime)
    logger.info("Freshness manifest seeded from current repo state")


async def sync_index(repo_path: str, vector_store) -> dict:
    """Scan for files changed since the last sync (mtime + content hash, with a
    settle window so mid-edit files aren't indexed half-written) and update the
    vector store incrementally — new files indexed, changed files have their
    old chunks replaced, removed files have their chunks deleted.
    """
    client = vector_store.client
    collection_name = config["qdrant"]["collection_name"]
    _ensure_payload_index(client, collection_name)

    manifest = await _get_manifest()
    current_files = set(get_source_files(repo_path))
    now = time.time()

    added, updated, deleted, skipped_unsettled = [], [], [], []

    for filepath in current_files:
        try:
            stat = os.stat(filepath)
        except FileNotFoundError:
            continue
        mtime = stat.st_mtime
        known = manifest.get(filepath)

        if known is not None and known["mtime"] == mtime:
            continue  # unchanged — cheapest possible check, no hashing needed

        if now - mtime < SETTLE_SECONDS:
            skipped_unsettled.append(filepath)
            continue  # too recently touched — might still be mid-edit, retry next scan

        content_hash = _hash_file(filepath)

        if known is None:
            n = _index_file(vector_store, filepath)
            if n:
                await _upsert_manifest(filepath, content_hash, mtime)
                added.append(filepath)
            continue

        if content_hash == known["hash"]:
            # touched but content identical (e.g. saved with no real edit) —
            # just record the new mtime so we stop re-checking this file's hash
            await _upsert_manifest(filepath, content_hash, mtime)
            continue

        _delete_file_chunks(client, collection_name, filepath)
        n = _index_file(vector_store, filepath)
        if n:
            await _upsert_manifest(filepath, content_hash, mtime)
        else:
            await _remove_from_manifest(filepath)
        updated.append(filepath)

    for filepath in manifest:
        if filepath not in current_files:
            _delete_file_chunks(client, collection_name, filepath)
            await _remove_from_manifest(filepath)
            deleted.append(filepath)

    if added or updated or deleted:
        logger.info(
            f"Freshness sync: {len(added)} added, {len(updated)} updated, {len(deleted)} deleted"
        )

    return {"added": added, "updated": updated, "deleted": deleted, "skipped_unsettled": skipped_unsettled}
