from services.security.crypto import envelope_decrypt, envelope_encrypt, inventory, is_encrypted


def test_inventory():
    inv = inventory()
    assert "postgres" in inv and is_encrypted("postgres")
    assert is_encrypted("api_tls")
def test_envelope():
    ct, digest = envelope_encrypt(b"hello", key=b"k"*16)
    assert envelope_decrypt(ct, b"k"*16, digest) == b"hello"
    try:
        envelope_decrypt(ct, b"wrong", digest)
        assert False
    except ValueError:
        pass
