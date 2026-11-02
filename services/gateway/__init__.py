"""services/gateway — route traffic and auth boundary.

A thin, dependency-free reverse-proxy/auth gateway that sits in front of the
Kotha GPT API. It is framework-agnostic (pure stdlib + optional Starlette/FastAPI
adapter) so it can run as a standalone process, a middleware, or a serverless
edge function.

Components:

- :class:`RouteTable` — registered upstream routes (host → upstream URL).
- :class:`AuthPolicy` — bearer-token / API-key policy.
- :class:`Gateway` — resolves a request to an upstream and enforces auth.
- :class:`ProxyGateway` — httpx-backed reverse proxy that forwards requests
  to the resolved upstream and returns the upstream response.
- :class:`RateLimiter` — simple per-key token-bucket rate limiter.
- :func:`asgi_app` — ASGI/Starlette-compatible gateway application.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, ClassVar


@dataclass(frozen=True)
class Route:
    """A single upstream route."""

    path_prefix: str
    upstream: str
    require_auth: bool = True


@dataclass
class RouteTable:
    """Ordered list of routes; first matching prefix wins."""

    routes: list[Route] = field(default_factory=list)

    def add(self, path_prefix: str, upstream: str, *, require_auth: bool = True) -> None:
        self.routes.append(Route(path_prefix, upstream, require_auth=require_auth))

    def resolve(self, path: str) -> Route | None:
        for route in self.routes:
            if path == route.path_prefix or path.startswith(route.path_prefix.rstrip("/") + "/"):
                return route
        return None


@dataclass
class AuthPolicy:
    """Bearer-token auth policy.

    ``api_token``: the single valid token (None = no auth enforced).
    ``local_mode``: when True and no token is configured, auth is bypassed.
    """

    api_token: str | None = None
    local_mode: bool = False

    def requires_auth(self) -> bool:
        return self.api_token is not None and not self.local_mode

    def check(self, authorization: str | None) -> None:
        if not self.requires_auth():
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise PermissionError("bearer token required")
        token = authorization.removeprefix("Bearer ").strip()
        if not token or not secrets.compare_digest(token, self.api_token):
            raise PermissionError("invalid bearer token")

    def authorize(self, authorization: str | None) -> None:
        """Alias for :meth:`check` for use by the gateway proxy."""
        self.check(authorization)


@dataclass
class GatewayRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes | None = None


@dataclass
class GatewayResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class Gateway:
    """Resolves requests to upstreams and enforces the auth policy."""

    def __init__(self, routes: RouteTable, auth: AuthPolicy) -> None:
        self.routes = routes
        self.auth = auth

    def route(self, request: GatewayRequest) -> tuple[Route | None, str | None]:
        """Return (route, upstream_url) or (None, None) if no route matches."""
        route = self.routes.resolve(request.path)
        if route is None:
            return None, None
        upstream = route.upstream.rstrip("/")
        suffix = request.path[len(route.path_prefix.rstrip("/")) :]
        url = upstream + suffix
        return route, url

    def authorize(self, request: GatewayRequest) -> None:
        self.auth.check(request.headers.get("authorization"))

    def handle(self, request: GatewayRequest) -> GatewayResponse:
        try:
            self.authorize(request)
        except PermissionError as exc:
            return GatewayResponse(
                status=401,
                headers={"content-type": "application/json"},
                body=b'{"error":"' + str(exc).encode() + b'"}',
            )

        route, _ = self.route(request)
        if route is None:
            return GatewayResponse(
                status=404,
                headers={"content-type": "application/json"},
                body=b'{"error":"no route for path"}',
            )

        # The gateway does not perform HTTP forwarding itself; callers use the
        # resolved ``url`` with their HTTP client of choice (httpx/urllib).
        # This keeps the gateway dependency-free and deployable anywhere.
        return GatewayResponse(
            status=200,
            headers={
                "content-type": "application/json",
                "x-kotha-gateway-route": route.path_prefix,
                "x-kotha-gateway-upstream": route.upstream,
            },
            body=(
                '{"ok":true,"upstream":"' + route.upstream + '","path":"' + request.path + '"}'
            ).encode(),
        )


# ---------------------------------------------------------------------------
# Rate limiter (per-key token bucket)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple per-key token-bucket rate limiter used by the gateway."""

    def __init__(self, max_requests: int = 60, window_sec: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, key: str = "global") -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.max_requests:
                retry = int(q[0] + self.window - now) + 1
                return False, max(retry, 1)
            q.append(now)
            return True, 0

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


# ---------------------------------------------------------------------------
# ProxyGateway — httpx-backed reverse proxy
# ---------------------------------------------------------------------------


class ProxyGateway:
    """HTTP reverse proxy that forwards requests to the resolved upstream.

    Uses :class:`httpx.AsyncClient` for forwarding. Headers that are
    hop-by-hop or gateway-managed are stripped/rewritten. A request ID is
    injected for traceability.
    """

    _HOP_BY_HOP: ClassVar[set[str]] = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }

    def __init__(
        self,
        routes: RouteTable,
        auth: AuthPolicy,
        rate_limiter: RateLimiter | None = None,
        timeout_sec: float = 30.0,
    ) -> None:
        self.routes = routes
        self.auth = auth
        self.rate_limiter = rate_limiter or RateLimiter()
        self.timeout_sec = timeout_sec

    def route(self, request: GatewayRequest) -> tuple[Route | None, str | None]:
        """Return (route, upstream_url) or (None, None) if no route matches."""
        route = self.routes.resolve(request.path)
        if route is None:
            return None, None
        upstream = route.upstream.rstrip("/")
        suffix = request.path[len(route.path_prefix.rstrip("/")) :]
        url = upstream + suffix
        return route, url

    def _client_headers(self, request: GatewayRequest) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in request.headers.items():
            if key.lower() in self._HOP_BY_HOP:
                continue
            headers[key] = value
        headers.setdefault("x-kotha-gateway", "1")
        headers.setdefault("x-request-id", request.headers.get("x-request-id", _new_id()))
        return headers

    async def forward(self, request: GatewayRequest) -> GatewayResponse:
        """Forward ``request`` to the resolved upstream and return its response."""
        # Auth
        try:
            self.auth.authorize(request.headers.get("authorization"))
        except PermissionError as exc:
            return GatewayResponse(
                status=401,
                headers={"content-type": "application/json"},
                body=b'{"error":"' + str(exc).encode() + b'"}',
            )

        # Rate limit
        authz = request.headers.get("authorization", "")
        key = "global"
        if authz.startswith("Bearer "):
            key = authz.removeprefix("Bearer ").strip() or "global"
        allowed, retry_after = self.rate_limiter.is_allowed(key)
        if not allowed:
            return GatewayResponse(
                status=429,
                headers={
                    "content-type": "application/json",
                    "retry-after": str(retry_after),
                },
                body=b'{"error":"rate limit exceeded"}',
            )

        # Route
        route, url = self.route(request)
        if route is None:
            return GatewayResponse(
                status=404,
                headers={"content-type": "application/json"},
                body=b'{"error":"no route for path"}',
            )

        import httpx

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                upstream_resp = await client.request(
                    method=request.method,
                    url=url,
                    headers=self._client_headers(request),
                    content=request.body,
                )
        except httpx.ConnectError as exc:
            return GatewayResponse(
                status=502,
                headers={"content-type": "application/json"},
                body=('{"error":"upstream unreachable: ' + str(exc) + '"}').encode(),
            )
        except httpx.TimeoutException as exc:
            return GatewayResponse(
                status=504,
                headers={"content-type": "application/json"},
                body=('{"error":"upstream timeout: ' + str(exc) + '"}').encode(),
            )

        upstream_headers: dict[str, str] = {}
        for key, value in upstream_resp.headers.items():
            if key.lower() in self._HOP_BY_HOP:
                continue
            upstream_headers[key] = value
        upstream_headers["x-kotha-gateway-proxied"] = "1"
        upstream_headers["x-kotha-gateway-upstream"] = route.upstream

        return GatewayResponse(
            status=upstream_resp.status_code,
            headers=upstream_headers,
            body=upstream_resp.content,
        )


def _new_id() -> str:
    import uuid

    return f"gw_{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# ASGI application
# ---------------------------------------------------------------------------


async def asgi_app(scope: dict[str, Any], receive: Callable, send: Callable) -> None:
    """Minimal ASGI application for the gateway.

    Supports ``http`` and ``lifespan`` scopes. Routes through
    :class:`ProxyGateway` with the default route table. Configure the API token
    via the ``KOTHAGPT_API_TOKEN`` environment variable.
    """
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
        return

    if scope["type"] != "http":
        return

    import os

    from . import AuthPolicy, ProxyGateway

    token = os.getenv("KOTHAGPT_API_TOKEN")
    local_mode = os.getenv("KOTHAGPT_LOCAL_MODE", "false").lower() == "true"
    auth = AuthPolicy(api_token=token, local_mode=local_mode)
    gateway = ProxyGateway(default_route_table(), auth)

    # Read body
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)

    headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])}
    request = GatewayRequest(
        method=scope["method"],
        path=scope["path"],
        headers=headers,
        body=body or None,
    )

    response = await gateway.forward(request)

    await send(
        {
            "type": "http.response.start",
            "status": response.status,
            "headers": [
                (k.encode("latin-1"), v.encode("latin-1")) for k, v in response.headers.items()
            ],
        }
    )
    await send({"type": "http.response.body", "body": response.body})


def default_route_table() -> RouteTable:
    table = RouteTable()
    table.add("/v1", "http://localhost:8000", require_auth=True)
    table.add("/health", "http://localhost:8000", require_auth=False)
    table.add("/v1/models", "http://localhost:8000", require_auth=False)
    return table


__all__ = [
    "AuthPolicy",
    "Gateway",
    "GatewayRequest",
    "GatewayResponse",
    "ProxyGateway",
    "RateLimiter",
    "Route",
    "RouteTable",
    "asgi_app",
    "default_route_table",
]
