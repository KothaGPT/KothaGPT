"""Tests for services.gateway — route traffic and auth boundary."""

from services.gateway import AuthPolicy, Gateway, GatewayRequest, RouteTable, default_route_table


def _req(method: str, path: str, headers: dict) -> GatewayRequest:
    return GatewayRequest(method=method, path=path, headers=headers, body=None)


def test_route_table_resolve():
    table = RouteTable()
    table.add("/v1", "http://localhost:8000")
    table.add("/health", "http://localhost:8000", require_auth=False)

    route = table.resolve("/v1/chat")
    assert route is not None
    assert route.path_prefix == "/v1"
    assert route.upstream == "http://localhost:8000"
    assert route.require_auth is True

    route = table.resolve("/health")
    assert route is not None
    assert route.require_auth is False

    assert table.resolve("/unknown") is None


def test_auth_policy_requires_auth():
    policy = AuthPolicy(api_token="secret")
    assert policy.requires_auth() is True

    policy.check("Bearer secret")  # should not raise

    try:
        policy.check(None)
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError for missing token")

    try:
        policy.check("Bearer wrong")
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError for wrong token")


def test_auth_policy_local_mode_bypasses():
    policy = AuthPolicy(api_token=None, local_mode=True)
    assert policy.requires_auth() is False
    policy.check(None)  # should not raise


def test_gateway_route_and_authorize():
    table = default_route_table()
    policy = AuthPolicy(api_token="secret")
    gw = Gateway(table, policy)

    # Missing auth on protected route
    resp = gw.handle(_req("GET", "/v1/chat", {}))
    assert resp.status == 401

    # Valid auth
    resp = gw.handle(_req("GET", "/v1/chat", {"authorization": "Bearer secret"}))
    assert resp.status == 200
    assert b"localhost:8000" in resp.body

    # No auth required for health
    gw_no_auth = Gateway(table, AuthPolicy(api_token=None, local_mode=True))
    resp = gw_no_auth.handle(_req("GET", "/health", {}))
    assert resp.status == 200

    # No route
    resp = gw.handle(_req("GET", "/unknown", {"authorization": "Bearer secret"}))
    assert resp.status == 404


def test_default_route_table():
    table = default_route_table()
    assert table.resolve("/v1/models") is not None
    assert table.resolve("/health") is not None
    health_route = table.resolve("/health")
    assert health_route.require_auth is False
