from services.security.secrets import SecretsManager, redact, scan_for_secrets


def test_vault():
    m = SecretsManager()
    m.set("API_KEY", "secret123")
    assert m.get("API_KEY") == "secret123"
    v = m.rotate("API_KEY", "newsecret")
    assert v == 2
    assert m.get("API_KEY") == "newsecret"

def test_redact():
    assert "***" in redact("api_key: sk-12345678901234567890")
    assert scan_for_secrets("my secret: abcdefgh12345678") != []

def test_no_leak():
    assert scan_for_secrets("hello world") == []
