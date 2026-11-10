"""Where sessions, jobs and the optional SAM2 config are stored."""
import os
from pathlib import Path


def default_home() -> Path:
    """Writable directory for runtime files.

    Override with ``XAL_TEMPORAL_HOME``. The default is the process working
    directory, never the package install tree and never the annotation folder.
    """
    raw = os.environ.get("XAL_TEMPORAL_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def worker_script() -> Path:
    """SAM2 worker entry point. Safe to launch with ``python -I``."""
    return Path(__file__).resolve().parent / "sam2_worker.py"
