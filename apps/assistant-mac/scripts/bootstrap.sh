#!/usr/bin/env bash
#
# Create (or refresh) the managed backend venv and install the assistant package.
#
# Thin-app model: the .app ships without Python; on first run the GUI calls this to
# stand up a venv using uv's own managed Python. Driven entirely by env vars so the
# GUI can parameterize every path:
#   UV              path to the uv binary                        (required)
#   VENV_DIR        where to create the venv                     (required)
#   WHEEL           assistant wheel to install into the venv     (required)
#   PYTHON_VERSION  uv-managed Python to use (default 3.12)
#   EXTRAS          wheel extras to install (default: all; "" for none)
#
set -euo pipefail

: "${UV:?UV (path to the uv binary) is required}"
: "${VENV_DIR:?VENV_DIR is required}"
: "${WHEEL:?WHEEL (path to the assistant wheel) is required}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# Floor: the backend needs >= 3.11 (pyproject requires-python; runtime uses tomllib).
# Keep in sync with requires-python in pyproject.toml. Reject a too-old override here,
# before downloading a managed interpreter that uv pip install would only refuse later.
MIN_PYTHON="3.11"
if [ "$(printf '%s\n%s\n' "$MIN_PYTHON" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_PYTHON" ]; then
    echo "error: backend requires Python >= ${MIN_PYTHON}, got ${PYTHON_VERSION}" >&2
    echo "       set a newer version under Settings > Backend runtime." >&2
    exit 1
fi

echo "==> Ensuring uv-managed Python ${PYTHON_VERSION}"
"$UV" python install "$PYTHON_VERSION"

# Create the venv only when missing — an update reuses the existing one (keeping its
# already-installed heavy deps like mlx-lm) and just reinstalls the assistant package.
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "==> Creating venv at ${VENV_DIR}"
    "$UV" venv "$VENV_DIR" --python "$PYTHON_VERSION"
else
    echo "==> Reusing existing venv at ${VENV_DIR}"
fi

# uv-created venvs ship without pip. Seed it (new or reused venv) so the in-app
# "Managed tools" installer can fall back to `python -m pip` when a uv binary isn't on
# the GUI-spawned backend's (minimal) PATH. Best effort — never block bootstrap on it.
"${VENV_DIR}/bin/python" -m ensurepip --upgrade || true

# Install ALL managed MLX tools by default (`all` extra = mlx,images,embeddings,vlm,audio,
# video) so every feature works out of the box; the native MLX backend in particular needs
# `mlx` (mlx-lm) or it reports UNAVAILABLE. `${EXTRAS-all}` defaults only when unset, so
# EXTRAS="" installs the bare wheel and EXTRAS="mlx,vlm" selects a subset. NOTE: the full
# set pulls heavy weight-runtime deps (mflux, mlx-audio, mlx-video) — a large first download.
EXTRAS="${EXTRAS-all}"
TARGET="$WHEEL"
[ -n "$EXTRAS" ] && TARGET="${WHEEL}[${EXTRAS}]"
# --reinstall-package assistant forces OUR package to reinstall even at the same version,
# so a rebuilt wheel's code changes (new tools, fixes) are actually picked up — uv would
# otherwise treat 0.1.0 as already satisfied and skip it. Heavy deps stay put, so an
# update is fast.
echo "==> Installing backend from ${TARGET}"
"$UV" pip install --python "${VENV_DIR}/bin/python" --reinstall-package assistant "$TARGET"

SERVER="${VENV_DIR}/bin/assistant-server"
if [ ! -x "$SERVER" ]; then
    echo "bootstrap failed: ${SERVER} not found after install" >&2
    exit 1
fi
# Record the installed wheel's hash so the app can detect a newer .app shipping a
# different wheel and trigger a reinstall (same version, changed code).
shasum -a 256 "$WHEEL" | awk '{print $1}' > "${VENV_DIR}/.assistant-wheel-sha" || true
echo "==> Ready: ${SERVER}"
