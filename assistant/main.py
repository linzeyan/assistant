"""FastAPI app factory + entry point.

Lifecycle: the SwiftUI app spawns this backend; this backend (in turn) connects to
or spawns the omlx model server. One supervision chain, torn down in reverse on exit.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from assistant.agent.compaction import CompactionManager
from assistant.agent.llm_client import AsyncLLM
from assistant.agent.hooks import HookRegistry
from assistant.agent.loop import AgentLoop
from assistant.downloads import DownloadManager
from assistant.gateway import lifecycle as gateway_lifecycle
from assistant.agent.session import SessionStore
from assistant.agent.trace import TraceStore
from assistant.api import (
    routes_audio,
    routes_chat,
    routes_config,
    routes_downloads,
    routes_images,
    routes_memory,
    routes_models,
    routes_sessions,
    routes_setup,
    routes_skills,
    routes_status,
    routes_traces,
    routes_video,
    routes_vision,
)
from assistant.config import Settings, get_settings
from assistant.images.mlx_backend import MlxImageBackend
from assistant.memory.file_provider import FileMemoryProvider
from assistant.models.default_store import DefaultModelStore
from assistant.models.mlx_audio import MlxAudioBackend
from assistant.models.mlx_embeddings import MlxEmbeddingBackend
from assistant.models.mlx_video import MlxVideoBackend
from assistant.models.mlx_vlm import MlxVLMBackend
from assistant.models.service import ModelService
from assistant.skills.discovery import SkillStore
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover, Rule
from assistant.tools.base import ToolContext

log = logging.getLogger("assistant")
req_log = logging.getLogger("assistant.request")

# High-frequency liveness polls from the GUI — logged at DEBUG so they don't drown the
# log with one-line-per-second noise (the user's complaint about the raw uvicorn access
# log). Everything else is logged at INFO with a request id, status, and duration.
_QUIET_PATHS = frozenset({"/status", "/downloads", "/preflight"})

# Bundled seed skills ship at the repo root, alongside the `assistant` package.
_BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _build_model_service(settings: Settings):
    """Construct the configured model backend behind the ``ModelService`` seam.

    Returns ``(service, cleanup)``; ``cleanup`` is an async callable that releases
    backend-specific resources (an HTTP client for omlx, loaded engines for mlx).
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
        include_hf_cache=settings.hf_cache,
        extra_model_dirs=settings.extra_model_dirs,
    )

    async def cleanup() -> None:
        await service.stop()

    return service, cleanup


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings

    # Ensure the user-facing model dirs exist so a fresh default install doesn't present
    # a "missing" path (and downloads have somewhere to land). Best-effort, never fatal.
    for d in {settings.models_dir, settings.download_dir}:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    model_service, cleanup = _build_model_service(settings)

    registry = build_registry()
    approver = PolicyApprover(settings.approval_required)
    skills = SkillStore(
        dirs=[_BUNDLED_SKILLS_DIR, settings.skills_dir], user_dir=settings.skills_dir
    )
    skills.scan()
    embedder = (
        MlxEmbeddingBackend(model=settings.embed_model)
        if settings.embed_memory
        else None
    )
    memory = FileMemoryProvider(settings.memory_dir, embedder=embedder)
    images = MlxImageBackend(
        settings.images_dir,
        model=settings.image_model,
        edit_quantize=settings.image_edit_quantize,
        width=settings.image_default_width,
        height=settings.image_default_height,
        steps=settings.image_default_steps,
    )
    vision = MlxVLMBackend(model=settings.vlm_model)
    audio = MlxAudioBackend(
        settings.audio_dir,
        stt_model=settings.stt_model,
        tts_model=settings.tts_model,
    )
    video = MlxVideoBackend(
        settings.video_dir,
        model=settings.video_model,
        checkpoint=settings.video_checkpoint,
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
    )

    app.state.model_service = model_service
    # Backend-authoritative default chat model (seeded from config), shared by the API and the
    # Telegram gateway so the GUI's "Default" applies to both. See models/default_store.py.
    app.state.default_model_store = DefaultModelStore(
        settings.home_dir / "default_model.json", seed=settings.default_model
    )
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
    download_manager = DownloadManager(
        target_dir=settings.download_dir,
        state_path=settings.home_dir / "downloads.json",
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
        compaction=compaction,
        max_output_tokens=settings.max_output_tokens,
        approval_rules=approval_rules,
        approval_ask_once=settings.approval_ask_once,
        hooks=hooks,
        trace_store=trace_store,
    )

    # Start the model backend. Non-fatal: a missing backend (no omlx / no mlx-lm)
    # still serves so the GUI can render and tell the user how to enable it.
    app.state.omlx_status = await model_service.start()

    gateway, app.state.telegram_error = await gateway_lifecycle.build_and_start(settings, app)
    app.state.telegram = gateway
    # Resume any download interrupted by the last shutdown (the hub continues partial files).
    await download_manager.resume_incomplete()
    try:
        yield
    finally:
        await download_manager.aclose()
        gw = getattr(app.state, "telegram", None)
        if gw is not None:
            await gw.stop()
        await cleanup()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="assistant-backend", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or get_settings()

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
    app.include_router(routes_images.router)
    app.include_router(routes_vision.router)
    app.include_router(routes_audio.router)
    app.include_router(routes_video.router)
    return app


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
    uvicorn.run(
        create_app(settings),
        host=settings.backend_host,
        port=settings.backend_port,
        # Per-request logging is handled by our middleware (with ids + timings), so disable
        # uvicorn's bare access log — it was the low-value spam in backend.out.log.
        access_log=False,
    )
