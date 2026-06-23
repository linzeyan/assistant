from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["status"])


@router.get("/status")
async def status(request: Request):
    svc = request.app.state.model_service
    st = svc.status
    return {
        "backend": "ok",
        "model_backend": request.app.state.settings.model_backend,
        # Key kept as "omlx" for GUI back-compat; for the native backend it carries
        # the in-process MLX status (state="local").
        "omlx": {
            "state": st.state.value if st else "unknown",
            "detail": st.detail if st else "not started",
            "base_url": st.base_url if st else None,
            "reachable": await svc.reachable(),
        },
    }
