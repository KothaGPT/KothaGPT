from services.security.pii_guard import detect_pii, mask_pii


def test_pii():
    assert detect_pii("test@example.com")["email"]
    assert "EMAIL_REDACTED" in mask_pii("email test@example.com")
    assert not detect_pii("hello world")
    assert mask_pii("hello", policy="drop") == "hello"
    assert mask_pii("call 123-45-6789", policy="drop") == "[REDACTED PII]"
