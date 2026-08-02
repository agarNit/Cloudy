import os
import warnings
import yaml
from pathlib import Path

# Set as early as possible — huggingface_hub/transformers read these once, the
# first time they're imported, and print progress bars / an "unauthenticated
# requests" notice straight to stderr, bypassing Python's logging module
# entirely. cloudy.config has no internal dependencies of its own, so it's
# reliably the first cloudy module imported (directly or transitively) by
# every entry point — main.py, the eval scripts — making this the one place
# that's actually guaranteed to run before whatever else first pulls in
# huggingface_hub. setdefault so a user can still opt into verbose output.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# sentence-transformers tries to re-enable progress bars on model load; disabling
# them above (intentionally) makes that attempt raise this UserWarning every time.
# Expected, harmless — not worth showing on every startup.
warnings.filterwarnings("ignore", message="Cannot enable progress bars.*")


def load_config() -> dict:
   """Load settings from config.yaml at the project root."""
   return yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())


config = load_config()
