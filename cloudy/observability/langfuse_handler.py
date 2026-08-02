from functools import lru_cache

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_langfuse_handler() -> CallbackHandler:
    """Reuses a single handler/client across calls, per Langfuse's own guidance.

    Credentials come from LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST env
    vars only — never hardcoded — so this works the same way for every user of this
    project: each person sets up their own Langfuse account (cloud or self-hosted)
    and only sees their own traces. If those env vars aren't set, the handler
    degrades to a silent no-op (verified directly) rather than raising, so
    observability stays fully optional, same as LangSmith already is.
    """
    return CallbackHandler()


def flush_langfuse() -> None:
    """Call before process exit — Langfuse batches spans and sends them on an
    interval, so a short-lived process (the eval runner) or an abrupt CLI exit can
    lose the most recent traces entirely without an explicit flush. Never raises:
    a flush problem shouldn't block cloudy from actually shutting down.
    """
    try:
        get_client().flush()
    except Exception as e:
        logger.warning(f"Langfuse flush failed: {e}")
