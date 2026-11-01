
from services.security.sandbox import Sandbox


def test_sandbox_blocks():
    s = Sandbox(allow_network=False)
    assert s.check_escape("cat /etc/passwd")
    assert s.check_escape("curl http://evil")
    assert not s.check_escape("echo hello")

def test_sandbox_run():
    s = Sandbox(cpu_sec=2, allow_network=False)
    r = s.run(["echo", "hello"])
    assert r.returncode == 0
    assert "hello" in r.stdout
    # Ensure teardown
    assert s._tmp is None or not s._tmp.exists()

def test_network_blocked():
    s = Sandbox(allow_network=False)
    try:
        s.run(["curl", "http://example.com"])
        assert False
    except PermissionError:
        pass
