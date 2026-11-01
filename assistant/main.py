"""FastAPI app factory + entry point.

Lifecycle: the SwiftUI app spawns this backend; this backend (in turn) connects to
or spawns the omlx model server. One supervision chain, torn down in reverse on exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import socket
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from assistant.agent.compaction import CompactionManager
from assistant.agent.llm_client import AsyncLLM
from assistant.agent.hooks import HookRegistry
from assistant.agent.loop import AgentLoop
from assistant.downloads import DownloadManager, hub_env
from assistant.gateway import lifecycle as gateway_lifecycle
from assistant.agent.session import InvalidSessionId, SessionStore
from assistant.agent.trace import TraceStore
from assistant.api import (
    access,
    routes_anthropic,
    routes_audio,
    routes_chat,
    routes_config,
    routes_downloads,
    routes_images,
    routes_logs,
    routes_memory,
    routes_models,
    routes_openai,
    routes_sessions,
    routes_setup,
    routes_skills,
    routes_status,
    routes_traces,
    routes_video,
    routes_vision,
)
from assistant import config as assistant_config
from assistant.config import Settings, get_settings, write_private
from assistant.images.mlx_backend import MlxImageBackend
from assistant.memory.file_provider import FileMemoryProvider
from assistant.agent.fusion import FusionEngine
from assistant.models.default_store import DefaultModelStore
from assistant.models.per_model_store import PerModelStore
from assistant.models.mlx_audio import MlxAudioBackend
from assistant.models.mlx_discovery import (
    discover_models,
    is_video_checkpoint,
    smallest_of_kind,
)
from assistant.models.mlx_embeddings import MlxEmbeddingBackend
from assistant.models.mlx_video import MlxVideoBackend
from assistant.models.mlx_vlm import MlxVLMBackend
from assistant.models.service import ModelService
from assistant.skills.discovery import SkillStore
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover, Rule
from assistant.tools.base import ToolContext
from assistant.tools.output_bounding import gc_spill_dir

log = logging.getLogger("assistant")
req_log = logging.getLogger("assistant.request")

# High-frequency liveness polls from the GUI — logged at DEBUG so they don't drown the
# log with one-line-per-second noise (the user's complaint about the raw uvicorn access
# log). Everything else is logged at INFO with a request id, status, and duration.
_QUIET_PATHS = frozenset({"/status", "/downloads", "/preflight"})

# Bundled seed skills ship at the repo root, alongside the `assistant` package.
_BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _build_model_service(settings: Settings, per_model=None, fusion=None):
    """Construct the configured model backend behind the ``ModelService`` seam.

    Returns ``(service, cleanup)``; ``cleanup`` is an async callable that releases
    backend-specific resources (an HTTP client for omlx, loaded engines for mlx).
    ``per_model`` (a PerModelStore) supplies per-model generation overrides to the mlx backend;
    ``fusion`` (a FusionEngine) surfaces the virtual "fusion" panel+judge model.
    """
    if settings.model_backend == "omlx":
        from assistant.models.omlx_client import OmlxClient
        from assistant.models.omlx_subprocess import OmlxProcess
        from assistant.models.service import OmlxModelService

        client = OmlxClient(settings.omlx_base_url, api_key=settings.omlx_api_key)
        process = OmlxProcess(
            client,
            host=settings.omlx_host,
            port=settings.omlx_port,
            models_dir=settings.models_dir,
            omlx_bin=settings.omlx_bin,
            autostart=settings.omlx_autostart,
        )
        service: ModelService = OmlxModelService(client, process)

        async def cleanup() -> None:
            await service.stop()
            await client.aclose()

        return service, cleanup

    from assistant.models.mlx_service import MlxModelService

    service = MlxModelService(
        models_dir=settings.models_dir,
        max_loaded=settings.max_loaded_models,
        max_concurrent=settings.mlx_max_concurrent,
        mem_ceiling_gb=settings.mem_ceiling_gb,
        include_hf_cache=settings.hf_cache,
        extra_model_dirs=settings.extra_model_dirs,
        per_model=per_model,
        fusion=fusion,
    )

    async def cleanup() -> None:
        await service.stop()

    return service, cleanup


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    # Startup timing (task: "backend start/restart is slow"). Static profiling showed the Python
    # side is fast (import ~0.2s, discovery ~0.01s, mlx-lm is lazy), so the cost — if any — is in
    # the async steps below (model backend start / Telegram gateway network handshake / download
    # resume). Log each phase's wall time so a real restart names the culprit instead of guessing.
    _boot = time.monotonic()

    def _phase(label: str, since: float) -> float:
        now = time.monotonic()
        log.info("startup: %s took %.2fs", label, now - since)
        return now

    # Ensure the user-facing model dirs exist so a fresh default install doesn't present
    # a "missing" path (and downloads have somewhere to land). Best-effort, never fatal.
    for d in {settings.models_dir, settings.download_dir, settings.workspace_dir}:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # Per-model generation overrides (oMLX-style), shared by the mlx backend (applied at chat
    # time) and the /models/{id}/settings API (read/write).
    per_model_store = PerModelStore(settings.home_dir / "per_model_settings.json")
    # Fusion engine (panel+judge virtual model); persisted config, editable via PUT /fusion.
    fusion = FusionEngine(
        settings.home_dir / "fusion.json",
        enabled=settings.fusion_enabled,
        panel=settings.fusion_panel,
        judge=settings.fusion_judge,
    )
    model_service, cleanup = _build_model_service(
        settings, per_model=per_model_store, fusion=fusion
    )

    registry = build_registry()
    approver = PolicyApprover(settings.approval_required)
    # Bundled, then user, then (optionally) a project/team dir scanned LAST so it wins on a slug
    # collision (E1). Capabilities for the requires gate are set further down, once the media
    # backends are built.
    _project_skill_dirs = (
        [settings.project_skills_dir] if settings.project_skills_dir else []
    )
    skills = SkillStore(
        dirs=[_BUNDLED_SKILLS_DIR, settings.skills_dir, *_project_skill_dirs],
        user_dir=settings.skills_dir,
    )
    skills.scan()
    for _slug, _overridden in skills.shadows().items():
        # Surface silent overrides (E1 shadow audit) so a project skill masking a user/bundled one
        # of the same name is diagnosable rather than a mystery.
        log.info("skill %r shadows definition(s) in %s", _slug, ", ".join(_overridden))
    # Prune stale spilled tool-output files (S15/H8) once at startup — the dir grows unbounded
    # otherwise. Best-effort; a sweep failure must never block bringing the backend up.
    _gc_removed, _gc_freed = gc_spill_dir(
        settings.tool_output_dir, max_age_days=settings.tool_output_retention_days
    )
    if _gc_removed:
        log.info("spill GC: removed %d file(s), freed %d bytes", _gc_removed, _gc_freed)
    embedder = (
        MlxEmbeddingBackend(model=settings.embed_model)
        if settings.embed_memory
        else None
    )
    memory = FileMemoryProvider(settings.memory_dir, embedder=embedder)
    # One discovery pass to default each category to its smallest (most memory-frugal) discovered
    # model rather than hardcoding a model name — so the defaults follow what the user actually
    # has, and a tiny default never OOMs on start. The user can still switch per category.
    discovered = discover_models(
        settings.models_dir,
        include_hf_cache=settings.hf_cache,
        extra_dirs=settings.extra_model_dirs,
    )

    images = MlxImageBackend(
        settings.images_dir,
        edit_quantize=settings.image_edit_quantize,
        width=settings.image_default_width,
        height=settings.image_default_height,
        steps=settings.image_default_steps,
    )
    # Default image model: an explicit image_model is matched as an id substring; otherwise the
    # smallest discovered image checkpoint. Resolved to an on-disk path so it routes to the mlxgen
    # CLI (generates in seconds). Unset + none discovered → the first generate fails loud with
    # "no image model selected" rather than silently pulling a multi-GB FLUX download.
    _images_found = [m for m in discovered if m.kind == "image"]
    if _images_found:
        pref = settings.image_model
        chosen = next(
            (m for m in _images_found if pref and pref in m.id), None
        ) or smallest_of_kind(discovered, {"image"})
        if chosen:
            images.set_model(str(chosen.path))
    vision = MlxVLMBackend(model=settings.vlm_model)
    audio = MlxAudioBackend(
        settings.audio_dir,
        stt_model=settings.stt_model,
        tts_model=settings.tts_model,
    )
    # Default video checkpoint: the smallest loadable mlx-video checkpoint when none configured.
    video_checkpoint = settings.video_checkpoint
    if video_checkpoint is None:
        _videos = [
            m for m in discovered if m.kind == "video" and is_video_checkpoint(m.path)
        ]
        if _videos:
            video_checkpoint = min(_videos, key=lambda m: m.size_bytes).path
    video = MlxVideoBackend(
        settings.video_dir,
        model=settings.video_model,
        checkpoint=video_checkpoint,
    )
    ctx = ToolContext(
        cwd=settings.workspace_dir,
        skills=skills,
        memory=memory,
        images=images,
        vision=vision,
        audio=audio,
        video=video,
        output_spill_dir=settings.tool_output_dir,
        # The backend's own output and the Telegram inbox: the user's files, readable by the
        # media tools without a prompt even though they sit outside the workspace.
        read_roots=(
            settings.images_dir,
            settings.video_dir,
            settings.audio_dir,
            settings.tool_output_dir,
            settings.telegram_inbox,
        ),
    )
    # Capability set for the skill requires-gate (E2): which optional media backends are installed.
    # Set now that the services exist; it's process-stable, so the gated skills index stays a
    # byte-stable cacheable prefix (S2/S3) — a skill requiring an absent backend is dropped here.
    skills.set_capabilities(
        {
            name
            for name, svc in (
                ("vision", vision),
                ("audio", audio),
                ("video", video),
                ("images", images),
            )
            if svc is not None and svc.available()
        }
    )

    app.state.model_service = model_service
    # Backend-authoritative default chat model, shared by the API and the Telegram gateway so the
    # GUI's "Default" applies to both. Seeded from config, or — when unset — the smallest
    # chattable model discovered, so a fresh install starts on its lightest model instead of an
    # arbitrary first one. The seed is only a fallback; a user's saved "Default" still wins.
    chat_seed = settings.default_model
    if not chat_seed:
        smallest_chat = smallest_of_kind(discovered, {"llm", "vlm"})
        chat_seed = smallest_chat.id if smallest_chat else None
    app.state.default_model_store = DefaultModelStore(
        settings.home_dir / "default_model.json", seed=chat_seed
    )
    app.state.per_model_store = per_model_store
    app.state.fusion = fusion
    app.state.skills = skills
    app.state.memory = memory
    app.state.images = images
    app.state.vision = vision
    app.state.audio = audio
    app.state.video = video
    app.state.sessions = SessionStore(settings.sessions_dir)
    # Shared registries: interactive-approval futures keyed by token.
    app.state.pending_approvals = {}
    # Model download manager (N17): progress / cancel / resume-after-restart / retry,
    # persisted to downloads.json under the data dir.
    # Download tunables (N50): extra hub env + concurrency, GUI-editable (config↔Settings). See
    # hub_env — HF_HUB_DISABLE_XET is the big one (Xet was measured throttling to a few KB/s).
    download_manager = DownloadManager(
        target_dir=settings.download_dir,
        state_path=settings.home_dir / "downloads.json",
        env=hub_env(
            disable_xet=settings.hf_hub_disable_xet,
            download_timeout=settings.hf_hub_download_timeout,
        ),
        max_workers=settings.hf_download_max_workers,
    )
    app.state.download_manager = download_manager
    # Managed-tool installs keyed by feature name.
    app.state.installs = {}
    app.state.install_tasks = {}
    llm = AsyncLLM(model_service)
    compaction = (
        CompactionManager(
            llm,
            context_window_fallback=settings.compaction_context_window,
            reserve_tokens=settings.compaction_reserve_tokens,
            keep_recent_tokens=settings.compaction_keep_recent_tokens,
        )
        if settings.compaction_enabled
        else None
    )
    app.state.compaction = compaction
    # S5: parse wildcard permission rules once; a malformed rule fails loud at startup
    # rather than silently mis-authorising a tool at call time.
    approval_rules = [Rule.from_dict(r) for r in settings.approval_rules]
    # P2 hook seam: empty registry exposed on app.state for in-process extensions to register.
    hooks = HookRegistry()
    app.state.hooks = hooks
    # P0 per-turn trace (spring2): records each turn so reliability failures are visible
    # (GET /sessions/{id}/turns, /turns/{id}). Off → memory-only store, no disk writes.
    trace_store = TraceStore(settings.trace_dir if settings.trace_enabled else None)
    app.state.trace_store = trace_store
    app.state.agent = AgentLoop(
        llm,
        registry,
        approver,
        ctx,
        max_iters=settings.max_tool_iters,
        compaction=compaction,
        max_output_tokens=settings.max_output_tokens,
        turn_timeout_s=settings.turn_timeout_s,
        approval_rules=approval_rules,
        approval_ask_once=settings.approval_ask_once,
        hooks=hooks,
        trace_store=trace_store,
    )
    # N105: product-native subagent fan-out. Assigned onto the SHARED tool context after the
    # loop deps exist — tools read it per call, and the spawn_subagents schema gate keys off
    # its presence (subagent children get a context without it, so no recursion). Children
    # ride the N104 batch lane, decoding together on one model where the engine allows.
    from assistant.agent.subagents import SubagentRunner

    ctx.subagents = SubagentRunner(
        llm,
        registry,
        approval_rules=approval_rules,
        max_output_tokens=settings.max_output_tokens,
        turn_timeout_s=settings.turn_timeout_s,
    )

    _t = _phase("build services + discovery", _boot)
    # Start the model backend. Non-fatal: a missing backend (no omlx / no mlx-lm)
    # still serves so the GUI can render and tell the user how to enable it.
    app.state.omlx_status = await model_service.start()
    _t = _phase("model backend start", _t)

    # Telegram's start() does a ~2s network handshake to api.telegram.org and nothing the GUI needs
    # waits on it — so start it in the BACKGROUND instead of blocking boot (it was ~half of a ~4s
    # startup). It publishes app.state.telegram / telegram_error once connected; until then
    # status() simply reports "not running".
    app.state.telegram = None
    app.state.telegram_error = None

    async def _start_telegram() -> None:
        gw, err = await gateway_lifecycle.build_and_start(settings, app)
        app.state.telegram = gw
        app.state.telegram_error = err
        log.info("startup: telegram gateway connected in background")

    telegram_task = asyncio.create_task(_start_telegram())
    # Resume any download interrupted by the last shutdown (the hub continues partial files).
    await download_manager.resume_incomplete()
    _phase("download resume", _t)
    log.info(
        "startup: backend ready in %.2fs total (telegram connecting in background)",
        time.monotonic() - _boot,
    )
    try:
        yield
    finally:
        # Stop the background start if it's still mid-handshake, then stop a connected gateway.
        telegram_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await telegram_task
        await download_manager.aclose()
        gw = getattr(app.state, "telegram", None)
        if gw is not None:
            await gw.stop()
        await cleanup()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="assistant-backend", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or get_settings()
    # Per-process secret the desktop app reads from an owner-only file (written by run()) and
    # presents on every request — see access.py for why loopback alone isn't enough.
    app.state.client_token = secrets.token_urlsafe(32)

    # Registered before log_requests so it sits inside it: refusals still get their log line.
    @app.middleware("http")
    async def guard_access(request: Request, call_next):
        if not access.is_allowed(
            request, app.state.settings.api_token, app.state.client_token
        ):
            log.warning(
                "refused %s %s: client=%s host=%r origin=%r (no valid token, or a browser "
                "page from another site)",
                request.method, request.url.path,
                request.client.host if request.client else "?", request.headers.get("host"),
                request.headers.get("origin"),
            )
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        # The compat routes read the body with request.json() whatever its declared type. A
        # page can POST text/plain without a CORS preflight; requiring JSON forces one, which
        # this backend never answers.
        if request.method == "POST" and request.url.path.startswith("/v1/"):
            ctype = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                return JSONResponse(
                    {"detail": "Content-Type must be application/json"}, status_code=415
                )
        return await call_next(request)

    @app.exception_handler(InvalidSessionId)
    async def invalid_session_id(request: Request, exc: InvalidSessionId):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """One structured line per request (id + method + path + status + duration) so the
        log is actually useful for debugging — replaces uvicorn's bare access lines. The id
        is also returned as ``x-request-id`` so a client report can be tied to a log line."""
        rid = uuid.uuid4().hex[:8]
        request.state.request_id = rid
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            dur = (time.perf_counter() - started) * 1000
            req_log.exception(
                "req %s %s %s -> unhandled error %.0fms", rid, request.method,
                request.url.path, dur,
            )
            raise
        dur = (time.perf_counter() - started) * 1000
        path = request.url.path
        if response.status_code >= 500:
            level = logging.ERROR
        elif response.status_code >= 400:
            level = logging.WARNING
        elif path in _QUIET_PATHS:
            level = logging.DEBUG
        else:
            level = logging.INFO
        req_log.log(
            level, "req %s %s %s -> %d %.0fms", rid, request.method, path,
            response.status_code, dur,
        )
        response.headers["x-request-id"] = rid
        return response

    app.include_router(routes_status.router)
    app.include_router(routes_models.router)
    app.include_router(routes_downloads.router)
    app.include_router(routes_setup.router)
    app.include_router(routes_config.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_sessions.router)
    app.include_router(routes_traces.router)
    app.include_router(routes_skills.router)
    app.include_router(routes_memory.router)
    app.include_router(routes_logs.router)
    app.include_router(routes_images.router)
    app.include_router(routes_vision.router)
    app.include_router(routes_audio.router)
    app.include_router(routes_video.router)
    # OpenAI- / Anthropic-compatible shims: let external agents (Claude Code, OpenAI clients)
    # drive the local models as a raw chat backend, bypassing our AgentLoop (they bring their own).
    app.include_router(routes_openai.router)
    app.include_router(routes_anthropic.router)
    return app


def client_token_path() -> Path:
    # A fixed XDG location rather than a setting: the desktop app has to find it before it
    # can ask the backend anything (Bootstrap.dataHome() mirrors XDG_DATA_DIR).
    return assistant_config.XDG_DATA_DIR / "client-token"


def _secure_private_paths(settings: Settings) -> None:
    """Owner-only permissions on the data, log and config dirs and on config.toml. They hold
    conversations, memory, logs of tool calls and the bot/api tokens; an install from before
    this was enforced has them world-readable under the default umask."""
    config_dir = assistant_config.XDG_CONFIG_DIR
    for d in (settings.home_dir, settings.log_dir, config_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
            d.chmod(0o700)
        except OSError:
            log.warning("could not restrict permissions on %s", d, exc_info=True)
    cfg = config_dir / "config.toml"
    if cfg.is_file():
        cfg.chmod(0o600)


def _port_is_free(host: str, port: int) -> bool:
    """Whether uvicorn will be able to bind host:port — probed the way it binds (SO_REUSEADDR,
    so a restart's TIME_WAIT doesn't count as taken)."""
    try:
        family, kind, proto, _, addr = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )[0]
        with socket.socket(family, kind, proto) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(addr)
        return True
    except OSError:
        return False


def run() -> None:
    import uvicorn

    from assistant.logging_setup import configure_logging

    settings = get_settings()
    log_path = configure_logging(log_dir=settings.log_dir, level=settings.log_level)
    log.info(
        "assistant-server starting: host=%s port=%s backend=%s models_dir=%s log=%s",
        settings.backend_host,
        settings.backend_port,
        settings.model_backend,
        settings.models_dir,
        log_path,
    )
    tls = {}
    if settings.ssl_certfile and settings.ssl_keyfile:
        tls = {"ssl_certfile": str(settings.ssl_certfile), "ssl_keyfile": str(settings.ssl_keyfile)}
    if not access.is_loopback_bind(settings.backend_host):
        if not tls:
            msg = (
                f"backend_host={settings.backend_host} exposes the API beyond this machine "
                "without TLS: set ssl_certfile and ssl_keyfile, or bind 127.0.0.1"
            )
            log.error(msg)
            raise SystemExit(msg)
        if not settings.api_token:
            log.warning(
                "backend_host=%s exposes the API beyond this machine but api_token is unset: "
                "every non-local request will be refused until one is configured",
                settings.backend_host,
            )
    # Checked before the token file is written: a second instance that would only fail to bind
    # (`make run` while the app's backend is up) must not replace the running one's token —
    # that locks the desktop app out of a backend that is still serving.
    if not _port_is_free(settings.backend_host, settings.backend_port):
        msg = f"{settings.backend_host}:{settings.backend_port} is already in use"
        log.error(msg)
        raise SystemExit(msg)
    _secure_private_paths(settings)
    app = create_app(settings)
    write_private(client_token_path(), app.state.client_token)
    uvicorn.run(
        app,
        host=settings.backend_host,
        port=settings.backend_port,
        # Per-request logging is handled by our middleware (with ids + timings), so disable
        # uvicorn's bare access log — it was the low-value spam in backend.out.log.
        access_log=False,
        **tls,
    )
