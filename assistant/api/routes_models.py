from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["models"])


def _svc(request: Request):
    return request.app.state.model_service


@router.get("/models")
async def list_models(request: Request):
    svc = _svc(request)
    models = await svc.list_models()
    return {
        "models": [asdict(m) for m in models],
        "reachable": await svc.reachable(),
    }


@router.get("/models/default")
async def get_default_model(request: Request):
    return {"default": request.app.state.default_model_store.value}


@router.put("/models/default")
async def set_default_model(request: Request):
    body = await request.json()
    request.app.state.default_model_store.set(body.get("model"))
    return {"default": request.app.state.default_model_store.value}


@router.post("/models/{model_id:path}/load")
async def load_model(model_id: str, request: Request):
    try:
        await _svc(request).load(model_id)
    except Exception as exc:  # surface the reason as a clean 502, not a 500 stacktrace
        raise HTTPException(status_code=502, detail=f"load failed: {exc}") from exc
    return {"ok": True, "model": model_id}


@router.post("/models/{model_id:path}/unload")
async def unload_model(model_id: str, request: Request):
    try:
        await _svc(request).unload(model_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"unload failed: {exc}"
        ) from exc
    return {"ok": True, "model": model_id}


@router.delete("/models/{model_id:path}")
async def delete_model(model_id: str, request: Request):
    svc = _svc(request)
    if not hasattr(svc, "delete"):
        raise HTTPException(
            status_code=501, detail="this backend can't delete models"
        )
    try:
        await svc.delete(model_id)
    except ValueError as exc:  # unknown model / not deletable — a client error, not 5xx
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"delete failed: {exc}") from exc
    return {"ok": True, "model": model_id}
