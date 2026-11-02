import pathlib
import tempfile

from services.security.signing import sign_artifact, verify_artifact


def test_sign():
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "model.pt"
        p.write_bytes(b"fake weights")
        sign_artifact(p, key=b"k")
        assert verify_artifact(p, key=b"k")
        p.write_bytes(b"tampered")
        assert not verify_artifact(p, key=b"k")
