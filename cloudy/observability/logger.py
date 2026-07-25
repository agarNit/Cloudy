import logging
from pathlib import Path


_LOG_DIR = Path(".cloudy")
_LOG_DIR.mkdir(exist_ok=True)

# Logs go to a file, not the console — the REPL owns the terminal output.
# Root logger stays at WARNING to suppress noisy third-party logs (httpcore etc.)
logging.basicConfig(
   level=logging.WARNING,
   format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
   filename=str(_LOG_DIR / "cloudy.log"),
)


def get_logger(name: str) -> logging.Logger:
   # Our own loggers run at DEBUG — only third-party libraries are suppressed
   logger = logging.getLogger(name)
   logger.setLevel(logging.DEBUG)
   return logger
