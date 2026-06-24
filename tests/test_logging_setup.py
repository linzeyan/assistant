"""Backend file logging (#3). The point: a GUI-spawned backend has no console, so its logs
must land in a findable file — pin that they actually do, and that a bad level is tolerated.
"""

from __future__ import annotations

import logging

from assistant.logging_setup import configure_logging


def _with_clean_root(fn):
    """Run ``fn`` with the global root logger saved and restored, so configuring logging in
    one test never bleeds handlers/level into the rest of the suite."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        fn()
    finally:
        root.handlers, root.level = saved_handlers, saved_level


def test_app_logs_land_in_the_file(tmp_path):
    def body():
        path = configure_logging(log_dir=tmp_path / "logs", level="DEBUG")
        assert path == tmp_path / "logs" / "backend.log"
        assert logging.getLogger().level == logging.DEBUG
        logging.getLogger("assistant").info("hello-test-line")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "hello-test-line" in path.read_text(encoding="utf-8")

    _with_clean_root(body)


def test_invalid_level_falls_back_to_info(tmp_path):
    def body():
        configure_logging(log_dir=tmp_path / "logs", level="bogus")
        assert logging.getLogger().level == logging.INFO

    _with_clean_root(body)


def test_httpx_request_logs_are_quieted(tmp_path):
    # httpx's per-request INFO lines would otherwise bury the app's logs in backend.log.
    def body():
        configure_logging(log_dir=tmp_path / "logs", level="INFO")
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING

    _with_clean_root(body)
