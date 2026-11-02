from services.security.audit import AuditLog


def test_audit():
    log = AuditLog()
    log.log("alice", "read", "model", "allow")
    log.log("bob", "write", "kb", "deny")
    assert len(log.entries) == 2
    assert log.verify_chain()
    assert len(log.query(actor="alice")) == 1
    # tamper
    log.entries[0]["actor"] = "evil"
    assert not log.verify_chain()
