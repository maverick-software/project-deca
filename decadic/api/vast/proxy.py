"""Reverse proxy: forward agent traffic to the tunnelled remote when active.

When a Vast deployment is ``ready`` the controller exposes a local tunnel port.
This middleware transparently forwards ``/agents``, ``/agent/*`` and
``/environment*`` to that remote so every existing dashboard panel renders the
remote agent without changing its base URL. ``/vast/*`` and everything else is
always served locally. WebSockets bypass HTTP middleware and are unaffected
(the body connects to the brain locally on the box).
"""

from __future__ import annotations

import logging

from fastapi import Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

# Path prefixes proxied to the remote while a deployment is active.
_PROXY_PREFIXES = ("/agents", "/agent/", "/environment")
# Response/request headers we must not copy verbatim across the hop.
_DROP_REQUEST_HEADERS = {"host", "content-length", "connection", "transfer-encoding"}
_DROP_RESPONSE_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
}


def _should_proxy(path: str) -> bool:
    if path == "/agents" or path.startswith("/agent/") or path == "/agent":
        return True
    return path == "/environment" or path.startswith("/environment/")


class VastProxyMiddleware(BaseHTTPMiddleware):
    """Forward agent/environment requests to the active deployment's tunnel."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        controller = getattr(request.app.state, "vast_controller", None)
        base = controller.proxy_base() if controller is not None else None
        if base is None or not _should_proxy(request.url.path):
            return await call_next(request)
        return await self._forward(request, base)

    async def _forward(self, request: Request, base: str) -> Response:
        import httpx

        url = f"{base}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                upstream = await client.request(
                    request.method, url, content=body, headers=headers
                )
        except Exception as exc:  # noqa: BLE001 - surface as a gateway error
            logger.warning("vast_proxy_error path=%s err=%s", request.url.path, exc)
            return JSONResponse(
                {"detail": f"remote deployment unreachable: {exc}"}, status_code=502
            )
        # Drop hop-by-hop headers and any upstream CORS headers; the local CORS
        # middleware (outermost) re-adds the correct ones for this origin.
        resp_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in _DROP_RESPONSE_HEADERS and not k.lower().startswith("access-control-")
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=resp_headers,
            media_type=upstream.headers.get("content-type"),
        )


def install_vast_proxy(app) -> None:
    """Attach the proxy middleware to the FastAPI app."""
    app.add_middleware(VastProxyMiddleware)
