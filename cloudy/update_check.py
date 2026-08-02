import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from cloudy import __version__
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

_REPO = "agarNit/Cloudy"
_LATEST_RELEASE_URL = f"https://api.github.com/repos/{_REPO}/releases/latest"
_CACHE_PATH = Path.home() / ".cloudy" / "update_check.json"
_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
_REQUEST_TIMEOUT_SECONDS = 2

UPGRADE_COMMAND = f"pipx install --force git+https://github.com/{_REPO}.git"


def _version_tuple(v: str) -> tuple:
    parts = []
    for part in v.split("."):
        digits = "".join(c for c in part if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _fetch_latest_version() -> str | None:
    req = Request(
        _LATEST_RELEASE_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "cloudy-cli"},
    )
    try:
        with urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.load(resp)
        return data.get("tag_name", "").lstrip("v") or None
    except (URLError, TimeoutError, OSError, ValueError) as e:
        logger.debug(f"Update check failed: {e}")
        return None


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _save_cache(data: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data))
    except OSError as e:
        logger.debug(f"Could not write update-check cache: {e}")


def check_for_update() -> str | None:
    """Return the latest released version if it's newer than this install, else None.

    Hits the GitHub API at most once every 24h (cached to disk under ~/.cloudy/) so a
    normal launch doesn't pay for a network round trip. Never raises — a failed or slow
    check should never block or break a normal cloudy session, just skip the notice.
    """
    cache = _load_cache()
    latest = cache.get("latest_version")

    if time.time() - cache.get("checked_at", 0) > _CHECK_INTERVAL_SECONDS:
        fetched = _fetch_latest_version()
        latest = fetched or latest
        _save_cache({"checked_at": time.time(), "latest_version": latest})

    if latest and _version_tuple(latest) > _version_tuple(__version__):
        return latest
    return None
