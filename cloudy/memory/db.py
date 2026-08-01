from pathlib import Path

from cloudy.config import config


def db_path() -> str:
    """Shared SQLite file for all of cloudy's own memory tables — session
    bookkeeping, episodic session logs, and long-term facts/preferences.
    Separate from LangGraph's own checkpoint tables, which live in the same
    file but are managed entirely by AsyncSqliteSaver.
    """
    path = Path(config["memory"]["db_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
