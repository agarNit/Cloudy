import os
from functools import lru_cache

from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def qdrant_reachable() -> bool:
    """Check once per process whether the configured Qdrant instance is reachable.

    Cached (rather than re-checked on every index/retrieve call) so a down Qdrant
    costs one short connection attempt at startup, not one per retrieval.
    """
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    try:
        QdrantClient(url=url, api_key=api_key, timeout=3).get_collections()
        return True
    except Exception as e:
        logger.warning(
            f"Qdrant unreachable ({e}); falling back to the local Chroma store for this session"
        )
        return False
