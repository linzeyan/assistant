"""Backend logging configuration.

The app spawns the backend with no console attached (GUI launch), so without an explicit
file sink the backend's own logs vanish — which is exactly when you need them. This installs
a rotating file handler (plus a console handler, useful under ``make run``) on the root
logger, so every ``logging.getLogger("assistant")`` call lands in one findable file.

Called once from the server entry point (``run()``), NOT from ``create_app`` — the test
suite builds apps directly and must not spray log files into temp dirs. uvicorn keeps its
own handlers on the ``uvicorn.*`` loggers (``propagate=False``), so per-request access-log
spam stays out of this file; only the app's own logs land here. uvicorn's lifecycle/error
output still goes to its console, which the GUI tees to ``backend.out.log`` separately.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(*, log_dir: Path, level: str = "INFO") -> Path | None:
    """Install root file+console handlers. Returns the log file path, or ``None`` when a
    file sink couldn't be created (degrades to console-only rather than crashing startup)."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter = logging.Formatter(_FMT)

    handlers: list[logging.Handler] = []
    path: Path | None = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "backend.log"
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handlers.append(file_handler)
    except OSError:
        path = None  # no writable file sink (perms / read-only home) → console-only

    handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(formatter)
    # Replace (not append) so a second call never duplicates handlers.
    root.handlers = handlers
    # httpx/httpcore log every outbound request at INFO (web_search, fetch, hub calls), which
    # buries the app's own lines in backend.log. They stay useful at WARNING.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path
