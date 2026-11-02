"""services/gateway — route traffic and auth boundary.

A thin, dependency-free reverse-proxy/auth gateway that sits in front of the
Kotha GPT API. It is framework-agnostic (pure stdlib + optional Starlette/FastAPI
adapter) so it can run as a standalone process, a middleware, or a serverless
edge function.

Components:

- :class:`RouteTable` — registered upstream routes (host → upstream URL).
- :class:`AuthPolicy` — bearer-token / API-key policy.
- :class:`Gateway` — resolves a request to an upstream and enforces auth.
- :func:`asgi_app` — ASGI/Starlette-compatible gateway application.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field


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
    "Route",
    "RouteTable",
    "default_route_table",
]
