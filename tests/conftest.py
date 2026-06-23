"""Test isolation: never read the developer's real ~/.config/assistant/config.toml.

`Settings()` merges that file as a low-priority source, so a developer who set a
non-default host/port/path via the GUI would otherwise pollute tests (e.g. a saved
`backend_host = "0.0.0.0"` failing an assertion that expects the default). Point the
XDG dirs at a throwaway temp dir *before* assistant.config computes its module-level
paths, so tests always see pristine defaults.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_ISOLATED = Path(tempfile.mkdtemp(prefix="assistant-test-xdg-"))
os.environ["XDG_CONFIG_HOME"] = str(_ISOLATED / "config")
os.environ["XDG_DATA_HOME"] = str(_ISOLATED / "data")

# Belt-and-suspenders: if assistant.config was already imported (unusual — conftest
# loads first), repoint its cached module-level dirs to match the isolated env.
try:
    import assistant.config as _cfg

    _cfg.XDG_CONFIG_DIR = _cfg._xdg_dir("XDG_CONFIG_HOME", ".config")
    _cfg.XDG_DATA_DIR = _cfg._xdg_dir("XDG_DATA_HOME", ".local/share")
except Exception:
    pass
