"""FastAPI app factory + entry point.

Lifecycle: the SwiftUI app spawns this backend; this backend (in turn) connects to
or spawns the omlx model server. One supervision chain, torn down in reverse on exit.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from assistant.agent.llm_client import AsyncLLM
from assistant.agent.loop import AgentLoop
from assistant.agent.session import SessionStore
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
    routes_video,
    routes_vision,
)
from assistant.config import Settings, get_settings
from assistant.images.mlx_backend import MlxImageBackend
from assistant.memory.file_provider import FileMemoryProvider
from assistant.models.mlx_audio import MlxAudioBackend
from assistant.models.mlx_embeddings import MlxEmbeddingBackend
from assistant.models.mlx_video import MlxVideoBackend
from assistant.models.mlx_vlm import MlxVLMBackend
from assistant.models.service import ModelService
from assistant.skills.discovery import SkillStore
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover
from assistant.tools.base import ToolContext

log = logging.getLogger("assistant")

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
    )
    vision = MlxVLMBackend(model=settings.vlm_model)
    audio = MlxAudioBackend(
        settings.audio_dir,
        stt_model=settings.stt_model,
        tts_model=settings.tts_model,
    )
    video = MlxVideoBackend(settings.video_dir, model=settings.video_model)
    ctx = ToolContext(
        cwd=settings.workspace_dir,
        skills=skills,
        memory=memory,
        images=images,
        vision=vision,
        audio=audio,
        video=video,
    )

    app.state.model_service = model_service
    app.state.skills = skills
    app.state.memory = memory
    app.state.images = images
    app.state.vision = vision
    app.state.audio = audio
    app.state.video = video
    app.state.sessions = SessionStore(settings.sessions_dir)
    # Shared registries: interactive-approval futures keyed by token, and in-flight
    # / finished model downloads keyed by repo id.
    app.state.pending_approvals = {}
    app.state.downloads = {}
    app.state.download_tasks = {}
    # Managed-tool installs keyed by feature name.
    app.state.installs = {}
    app.state.install_tasks = {}
    app.state.agent = AgentLoop(AsyncLLM(model_service), registry, approver, ctx)

    # Start the model backend. Non-fatal: a missing backend (no omlx / no mlx-lm)
    # still serves so the GUI can render and tell the user how to enable it.
    app.state.omlx_status = await model_service.start()

    gateway = await _maybe_start_telegram(settings, app)
    app.state.telegram = gateway
    try:
        yield
    finally:
        if gateway is not None:
            await gateway.stop()
        await cleanup()


async def _maybe_start_telegram(settings: Settings, app: FastAPI):
    """Start the Telegram gateway if a token is configured. Failure is non-fatal:
    the backend (and desktop GUI) must keep working even if Telegram can't connect."""
    if not settings.telegram_token:
        return None
    from assistant.gateway.telegram import TelegramGateway

    gateway = TelegramGateway(
        token=settings.telegram_token,
        allowed_users=settings.telegram_allowed_users,
        agent=app.state.agent,
        sessions=app.state.sessions,
        model_service=app.state.model_service,
        default_model=settings.default_model,
        approval_required=settings.approval_required,
        audio=app.state.audio,
    )
    try:
        await gateway.start()
        return gateway
    except Exception as exc:
        # A bad token is common user misconfiguration — a clean one-liner beats a
        # scary traceback. The backend must keep serving regardless.
        log.warning("Telegram gateway failed to start: %s; continuing without it", exc)
        return None


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="assistant-backend", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or get_settings()
    app.include_router(routes_status.router)
    app.include_router(routes_models.router)
    app.include_router(routes_downloads.router)
    app.include_router(routes_setup.router)
    app.include_router(routes_config.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_sessions.router)
    app.include_router(routes_skills.router)
    app.include_router(routes_memory.router)
    app.include_router(routes_images.router)
    app.include_router(routes_vision.router)
    app.include_router(routes_audio.router)
    app.include_router(routes_video.router)
    return app


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.backend_host,
        port=settings.backend_port,
    )
