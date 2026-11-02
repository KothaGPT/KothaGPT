from services.security.abuse import AbuseDetector


def test_abuse():
    d = AbuseDetector(burst_threshold=3, window_sec=60)
    assert not d.record("t1")
    assert not d.record("t1")
    assert d.record("t1")  # 3rd hits threshold
    assert d.is_flagged("t1")
    d.review("t1", "allow")
    assert not d.is_flagged("t1")
