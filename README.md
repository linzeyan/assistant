# Assistant

A **local-first AI assistant for macOS (Apple Silicon)**. Everything runs on your own
machine: a Python backend serves local model inference, and a native SwiftUI app gives
you chat, model management, a tool-using coding agent, skills, long-term memory, and
image / vision / audio / video — no cloud model and no account; your data stays local.
An optional Telegram gateway lets you reach the same agent from your phone.

```
   SwiftUI app  ─┐
                 ├─ REST + SSE ─▶  Python backend (FastAPI)  ─▶  local models (MLX)
   Telegram      ─┘
```

---

## Highlights

- **Runs entirely on-device** — local inference on Apple Silicon via [MLX](https://github.com/ml-explore/mlx); loopback-only by default.
- **Native macOS app** — menu-bar app with streaming chat, model switching, and inline image/video previews. It connects to a running backend or spawns one for you.
- **Tool-using agent** — a think → tool → observe loop that can read/write/edit files, run shell commands, search code, and call media tools. Mutating actions are **approval-gated**: you approve or deny each one.
- **Skills** — drop-in `SKILL.md` capabilities the agent can discover, use, and even author itself (behind approval).
- **Long-term memory** — file-based memory the agent writes to and recalls; upgrades to semantic search when the embeddings extra is installed.
- **Multimodal** — generate images (FLUX), generate video (Wan / LTX), read images (VLM), and transcribe / synthesize speech — each an optional add-on that **degrades gracefully** when absent.
- **Persistent conversations** — chats are saved to disk and survive restarts; reopen or delete them from the app.
- **Telegram gateway** — talk to the agent from your phone, including voice notes (speech in, speech out).
- **Graceful by design** — a missing model backend or uninstalled tool reports a clear status and returns `503`; the backend never crashes.

---

## Requirements

- **macOS 14+ on Apple Silicon** (M-series). MLX inference is Apple-Silicon-only.
- **Python 3.11+**.
- [**uv**](https://docs.astral.sh/uv/) for environment management (the app can install it for you on first launch).
- **Xcode or Command Line Tools** to build the SwiftUI app (`swift build` works under Command Line Tools alone).

---

## Quick start (run from source)

```sh
make setup-mlx     # venv + native MLX inference (mlx-lm) — enough to chat
make app-run       # build + launch the SwiftUI app (auto-starts the backend)
```

Want every modality (image / vision / audio / video) in one go:

```sh
make install       # venv + ALL managed MLX tools
make run           # start just the backend on http://127.0.0.1:9981
```

You still need a model. Download one from the app's **Downloads** tab (or drop MLX
weights into `~/.local/share/assistant/models/`), then pick it in the **Chat** tab.

`make help` lists every target.

---

## Using the app

The app is organized into tabs (plus a first-run **Setup** flow):

| Tab | What it does |
|-----|--------------|
| **Chat** | Pick a model and talk. Replies stream live; assistant reasoning and tool calls are collapsed by default. Generated images/videos preview inline. Tools that change things prompt an **Approve / Deny** bar. Past conversations are saved — reopen them from the **Conversations** popover. |
| **Models** | See discovered models, load / unload / switch the active one (an LRU pool keeps memory in check), and delete weights. |
| **Downloads** | Fetch models from Hugging Face by repo id into your configured model directory, with live progress. |
| **Skills** | Browse, create, edit, import, and reload `SKILL.md` skills. Bundled skills are read-only; your own are editable. |
| **Memory** | Inspect, search, add, and remove long-term memory entries. |
| **Settings** | Repoint the models / downloads directories and other paths, choose the model backend, and (for a packaged build) override the `uv` / venv / Python paths. Path changes apply after a restart. |
| **Status** | Backend reachability, the active model backend, and a runtime preflight summary. |
| **Setup** | On a fresh machine, stands up the backend (ensures `uv`, creates a managed venv, installs the bundled wheel) and installs optional MLX tools on demand. |

If the backend becomes unreachable, an **offline banner** with a Retry button appears
until it comes back.

---

## Model backends

Inference sits behind a pluggable `ModelService` seam, selected by the `model_backend`
config field:

- **`mlx` (default)** — native, **in-process** inference with
  [`mlx-lm`](https://github.com/ml-explore/mlx-lm). No external server. Discovers models
  (local directories + optionally the Hugging Face cache), manages them through an LRU
  engine pool, and parses tool-call text (Hermes / Qwen / Mistral / Llama formats) back
  into structured calls — so the agent's coding and self-authoring loops work here.
  Install with `make setup-mlx`.
- **`omlx`** — an external [omlx](https://github.com/jundot/omlx) server reached via
  connect-or-spawn (installed through Homebrew, not PyPI: `make omlx`). OpenAI-compatible,
  including tool calling.

Either way the backend boots and degrades gracefully if the chosen backend is absent.

---

## Tools, skills & memory

The agent runs a **think → tool → observe** loop and streams typed events over SSE.
Available tool families:

- **Files** — read, write, edit, glob, grep within the workspace directory.
- **Shell** — run `bash` commands.
- **Web** — search the web (DuckDuckGo) and fetch pages as readable text.
- **Skills** — `skill_manage` lets the agent create / patch / archive its own `SKILL.md` skills.
- **Memory** — write and search long-term memory; relevant entries are prefetched each turn.
- **Media** — generate images / video, read images, transcribe audio, synthesize speech (when the matching extra is installed).

Anything that changes state — `write_file`, `edit_file`, `bash`, `skill_manage` — is
**approval-gated**. In the app you get an Approve/Deny bar; over Telegram, inline buttons.
Approval is on by default (`approval_required`).

---

## Multimodal (optional extras)

Each modality is a heavy, Apple-Silicon-only opt-in. Install only what you need; the
matching feature reports *unavailable* (never crashes) when its extra is missing.

| Feature | Extra | Make target | Backed by |
|---------|-------|-------------|-----------|
| Image generation | `images` | `make setup-images` | mflux (FLUX) |
| Semantic memory search | `embeddings` | `make setup-embeddings` | mlx-embeddings |
| Read images (vision) | `vlm` | `make setup-vlm` | mlx-vlm |
| Speech-to-text + text-to-speech | `audio` | `make setup-audio` | mlx-audio |
| Text-to-video | `video` | `make setup-video` | mlx-video (Wan / LTX) |

`make install` installs them all.

---

## Telegram gateway

Set `telegram_token` (and a `telegram_allowed_users` allowlist — **deny by default**)
and the gateway runs in-process with the backend. Each chat gets its own persisted
session; replies stream by editing a single message; approval-gated tools prompt with
inline buttons. Voice notes are transcribed, answered by the agent, and replied to as
speech (text-only when the audio extra isn't installed). A bad token or failed start is
non-fatal — the backend keeps serving.

---

## Configuration & paths

Config lives at `~/.config/assistant/config.toml` (overridable via `$XDG_CONFIG_HOME`).
Any field can also be set with an `ASSISTANT_`-prefixed environment variable
(e.g. `ASSISTANT_BACKEND_PORT=9000`). The Settings tab edits the common ones and writes
them back to the TOML file.

Key defaults:

| Field | Default | Notes |
|-------|---------|-------|
| `backend_host` / `backend_port` | `127.0.0.1` / `9981` | Set host to `0.0.0.0` to expose on the LAN. |
| `model_backend` | `mlx` | or `omlx` |
| `models_dir` / `download_dir` | `~/.local/share/assistant/models` | Downloads land where models are discovered. |
| `hf_cache` | `false` | Also surface the shared Hugging Face cache in the model list. |
| `approval_required` | `true` | Require human approval for mutating tools. |
| `workspace_dir` | process cwd | Where file/shell tools operate. |

Application data follows the XDG spec under `~/.local/share/assistant/`:
`models/`, `skills/`, `memory/`, `sessions/`, `images/`, `videos/`, `audio/`.
Model weights also resolve from the Hugging Face cache (`~/.cache/huggingface`).

---

## API quick checks

With the backend running:

```sh
curl -s http://127.0.0.1:9981/status | python3 -m json.tool
curl -s http://127.0.0.1:9981/models | python3 -m json.tool
curl -sN http://127.0.0.1:9981/chat -H 'content-type: application/json' \
  -d '{"model":"<model-id>","message":"hello"}'      # SSE stream
```

The backend speaks REST for commands and Server-Sent Events for streaming chat
(`session` / `assistant_delta` / `tool_call` / `approval_request` / `tool_result` /
`error` / `done`).

---

## Packaging the macOS app

```sh
make app-package                              # build + bundle + ad-hoc sign → dist/Assistant.app
CODESIGN_IDENTITY="Developer ID Application: You (TEAMID)" make app-package   # distributable
make app-notarize NOTARY_PROFILE=my-profile   # notarize + staple (after a Developer ID build)
```

The `.app` is **thin** — it ships *no Python*. `app-package` builds the backend **wheel**
(`uv build`) and bundles it with `bootstrap.sh` in `Contents/Resources/backend`. On first
launch the **Setup** screen ensures `uv`, creates a managed venv with a uv-managed Python,
installs the bundled wheel, and spawns the backend; optional MLX tools are then installed
on demand. Ad-hoc signing needs no Apple account (first launch: right-click → Open); a
distributable build needs a Developer ID identity and notarization.

CI (`.github/workflows/build.yml`, macOS runner) runs the tests, builds + ad-hoc-signs the
app, uploads `Assistant.app.zip` as an artifact, and attaches it to a GitHub Release on
`v*` tags.

---

## Development

```sh
make setup     # venv + dev deps only (light — enough to run the tests)
make test      # Python test suite (all model backends faked — no MLX install needed)
make app-build # compile the SwiftUI app (debug)
make app-test  # run the SwiftUI app's Swift unit tests (local; not in CI)
make app-run   # build + launch the app (auto-starts the backend)
```

The Python suite fakes every model backend, so it runs without any MLX install. The Swift
suite (pure-logic: DTO decoding, the SSE reducer, message parsing, client framing) uses
[swift-testing](https://github.com/swiftlang/swift-testing) and runs locally via
`make app-test`.

### Project layout

```
assistant/             Python backend (FastAPI)
  api/                 REST + SSE routes
  agent/               think→tool→observe loop, sessions, prompts
  models/              ModelService seam (mlx / omlx), engine pool, tool-call parsing
  tools/               self-registering agent tools + approval
  memory/              file-based / semantic memory providers
  gateway/             Telegram gateway
  setup/               preflight + managed-tool install
apps/assistant-mac/    native SwiftUI app (SwiftPM)
tests/                 Python tests (faked backends)
Makefile               install / run / build / package targets
```

---

## Status & roadmap

**Working today:** model switching + streaming chat behind a pluggable backend (native
MLX or omlx); the approval-gated tool agent; skills (incl. self-authoring); file-based and
semantic memory; image / vision / audio / video modalities; persistent conversations; the
Telegram gateway (incl. voice); the native macOS app; and thin-app packaging with a
self-standing managed venv.

**On the roadmap:** stable cacheable prompt prefixes (KV-cache reuse), automatic
conversation compression for long sessions, unified tool-output bounding, wildcard
allow/deny/ask permission rules, supervised backend auto-restart, and a dedicated
**Gateways** settings tab for runtime Telegram control.
