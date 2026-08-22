# Assistant

**Your own AI assistant, running entirely on your Mac.**

Assistant turns an Apple Silicon Mac into a private, full-featured AI workstation: a
native macOS app with streaming chat, an agent that can actually *do things* on your
machine (with your approval), long-term memory, skills, and image / vision / audio /
video generation — all powered by local [MLX](https://github.com/ml-explore/mlx) models.
**No cloud, no account, no API key. Your conversations never leave your Mac.**

```
   macOS app    ─┐
   Telegram     ─┼─▶  local backend  ─▶  local models (MLX, on-device)
   Claude Code  ─┘
```

---

## What makes it different

- **Truly local** — inference runs on-device via MLX; the backend listens on loopback
  only by default. Works on an airplane.
- **An agent, not just a chatbot** — a think → tool → observe loop that reads, writes,
  and edits files, runs shell commands, searches code and the web, and calls media
  tools. Every mutating action shows an **Approve / Deny** bar before it runs — you
  stay in control, and you can pre-authorise safe tools or block dangerous ones with
  allow/deny/ask rules.
- **It can drive Claude Code** — the backend speaks the Anthropic `/v1/messages` and
  OpenAI `/v1/chat/completions` APIs, so Claude Code and other coding agents can use
  your local models as their backend. See [below](#use-it-as-a-backend-for-claude-code).
- **Remembers you** — file-based long-term memory the agent writes to and recalls each
  turn; semantic search when the embeddings extra is installed.
- **Teachable** — drop-in `SKILL.md` skills the agent discovers, uses, and can even
  author itself (behind approval).
- **Multimodal** — generate images (FLUX) and video (Wan / LTX), read images, and
  transcribe / synthesize speech. Each modality is an optional add-on that degrades
  gracefully when absent.
- **In your pocket** — an optional Telegram gateway brings the same agent (including
  voice notes: speech in, speech out) to your phone.
- **Built to stay up** — long chats are compacted automatically, prompt prefixes are
  KV-cached for fast follow-up turns, conversations persist across restarts, and the
  app supervises the backend and restarts it if it dies.

---

## Get started

You need a **Mac with Apple Silicon (M-series) on macOS 14+**.

**Option A — download the app**: grab `Assistant.app` from
[Releases](../../releases) (when available). On first launch a **Setup** screen
installs everything the backend needs into a self-contained environment — no manual
Python setup.

**Option B — build from source** (needs [uv](https://docs.astral.sh/uv/) and Xcode
Command Line Tools):

```sh
make setup-mlx     # backend + native MLX inference — enough to chat
make app-run       # build + launch the macOS app (auto-starts the backend)
```

Then download a model: open the app's **Downloads** tab, fetch an MLX model from
Hugging Face (e.g. `mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit`), and pick it in
the **Chat** tab. That's it.

---

## Using the app

| Tab | What it does |
|-----|--------------|
| **Chat** | Pick a model and talk. Replies stream live; reasoning and tool calls are collapsed by default. Generated images/videos preview inline. Tools that change things prompt **Approve / Deny**. Past conversations are saved — reopen them from the **Conversations** popover. |
| **Models** | See discovered models, load / unload / switch the active one (an LRU pool keeps memory in check), tune per-model sampling, and delete weights. |
| **Downloads** | Fetch models from Hugging Face by repo id, with live progress. |
| **Skills** | Browse, create, edit, import, and reload `SKILL.md` skills. Bundled skills are read-only; your own are editable. |
| **Memory** | Inspect, search, add, and remove long-term memory entries. |
| **Settings** | Organised by concern — **Backend** (paths, model backend), **Agent** (approval, workspace), **Gateways** (Telegram token + allowlist, save & restart in place), **Advanced**. |
| **Status** | Backend reachability, the active model backend, and a runtime preflight summary. |

If the backend becomes unreachable, an offline banner with a Retry button appears
until it comes back.

---

## Use it as a backend for Claude Code

The backend exposes an Anthropic-compatible `POST /v1/messages` (streaming +
non-streaming + tools), so Claude Code can run against your local models:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:9981
export ANTHROPIC_API_KEY=local          # any value; the local backend ignores it
export ANTHROPIC_MODEL=Qwen3-Coder-30B-A3B-Instruct-8bit
claude
```

Model names are matched fuzzily against your local models (short id, basename, or
substring — no `mlx-community/` prefix needed). OpenAI-style clients work too, via
`GET /v1/models` + `POST /v1/chat/completions`.

Tip: pick a model that is good at tool calling (Coder/Instruct variants). The Models
tab marks known weak-at-tools models with ⚠️.

---

## Put it to work on a codebase (unattended)

`drive/` is a small toolkit for using the agent as a pair of hands on a real
repository: you decide every line of the change and write it into a *brief*, the
model places it, runs the build and the tests, and reports what happened.

```sh
cd drive
./check-anchors.py brief.md ~/git/project        # every anchor unique?
./drive.py --brief brief.md --workspace ~/git/project --thinking off --effort low
```

It ships the driver, an anchor pre-flight, a brief template, a script for proving
a new test actually fails when you break the code — and, more usefully, the
measured rules that make local models reliable at this: what to put in a brief,
why `--thinking off` is the setting that matters, and the three ways a turn goes
wrong while still looking like progress. See **[drive/README.md](drive/README.md)**.

---

## Talk to it from your phone (Telegram)

In **Settings → Gateways**, paste a bot token and an allowed-user allowlist
(**deny by default**) and hit *Save & (re)start* — no config files needed. Each chat
gets its own persisted session; replies stream by editing a single message;
approval-gated tools prompt with inline buttons; voice notes are transcribed, answered,
and replied to as speech when the audio extra is installed.

---

## Optional extras (multimodal)

Each modality is a heavy, Apple-Silicon-only opt-in. Packaged app: install them from
the **Setup** screen on demand. From source:

| Feature | Make target | Backed by |
|---------|-------------|-----------|
| Image generation | `make setup-images` | mflux (FLUX) |
| Text-to-video | `make setup-video` | mlx-video (Wan / LTX) |
| Read images (vision) | `make setup-vlm` | mlx-vlm |
| Speech-to-text + text-to-speech | `make setup-audio` | mlx-audio |
| Semantic memory search | `make setup-embeddings` | mlx-embeddings |

`make install` installs them all. A missing extra reports *unavailable* — it never
crashes the backend.

---

## Configuration

Everyday settings live in the app's **Settings** tab. Under the hood, config is
`~/.config/assistant/config.toml`; any field can also be set with an
`ASSISTANT_`-prefixed environment variable (e.g. `ASSISTANT_BACKEND_PORT=9000`).

Key defaults:

| Field | Default | Notes |
|-------|---------|-------|
| `backend_host` / `backend_port` | `127.0.0.1` / `9981` | Set host to `0.0.0.0` to expose on the LAN. |
| `model_backend` | `mlx` | Native in-process inference; or `omlx` for an external server. |
| `models_dir` / `download_dir` | `~/.local/share/assistant/models` | Downloads land where models are discovered. |
| `approval_required` | `true` | Require human approval for mutating tools. |
| `approval_rules` | (none) | Wildcard allow/deny/ask rules per tool + resource; first match wins. |
| `workspace_dir` | process cwd | Where file/shell tools operate. |

Application data follows the XDG spec under `~/.local/share/assistant/`
(`models/`, `skills/`, `memory/`, `sessions/`, `images/`, `videos/`, `audio/`).
Model weights also resolve from the Hugging Face cache.

---

## For developers

```sh
make setup     # venv + dev deps only (light)
make test      # Python test suite — all model backends faked, no MLX install needed
make app-test  # Swift unit tests (swift-testing; local)
make app-package   # build + bundle + ad-hoc sign → dist/Assistant.app
make help      # every target
```

```
assistant/             Python backend (FastAPI)
  api/                 REST + SSE routes, OpenAI/Anthropic compat shims
  agent/               think→tool→observe loop, sessions, compaction, prompts
  models/              ModelService seam (mlx / omlx), engine pool, tool-call parsing
  tools/               self-registering agent tools + approval
  memory/              file-based / semantic memory providers
  gateway/             Telegram gateway
  setup/               preflight + managed-tool install
apps/assistant-mac/    native SwiftUI app (SwiftPM)
drive/                 drive the agent through a coding brief, unattended
tests/                 Python tests (faked backends)
```

The `.app` is thin — it ships no Python. `make app-package` bundles the backend wheel;
first launch creates a managed venv and installs it. Ad-hoc signing needs no Apple
account (first launch: right-click → Open); a distributable build needs a Developer ID
(`CODESIGN_IDENTITY=… make app-package`, then `make app-notarize`). CI runs the tests,
builds the app, and attaches `Assistant.app.zip` to Releases on `v*` tags.

Backend API quick check with the server running:

```sh
curl -s http://127.0.0.1:9981/status | python3 -m json.tool
curl -sN http://127.0.0.1:9981/chat -H 'content-type: application/json' \
  -d '{"model":"<model-id>","message":"hello"}'      # SSE stream
```
