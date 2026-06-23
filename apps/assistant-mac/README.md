# assistant-mac

Native SwiftUI front-end for the local assistant backend. Built as a SwiftPM
executable so it compiles and runs under Command Line Tools (no full Xcode needed
during development); proper `.app` bundling + signing is a later packaging step.

## Run

Start the Python backend first (from the repo root):

```sh
uv run assistant-server          # http://127.0.0.1:9981
```

Then build/run the app:

```sh
cd apps/assistant-mac
swift build                      # compile check
swift run                        # launch the window + menu-bar item
```

The app connects to `http://127.0.0.1:9981` by default (change it in Settings). It
talks only to the backend's HTTP API — REST for commands, SSE for streaming chat.

## Screens

- **Chat** — streaming replies; tool calls inline; generated images/videos render
  inline (loaded off disk); approval-required tools prompt with Approve/Deny buttons.
- **Models** — list/switch/load/unload local models (native MLX or omlx backend).
- **Downloads** — fetch a HuggingFace model by repo id; live status while it pulls.
- **Skills** — browse `SKILL.md` skills, view bodies, reload.
- **Memory** — browse and search long-term memory.
- **Status** — backend + model-backend health.
- **Settings** — backend URL.

## Notes

- `BackendController` is `@MainActor`; it's the single source of UI state.
- The model layer / skills / memory / image generation all live in the Python
  backend — this app is a thin client over it.
