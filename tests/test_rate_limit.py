from services.api.rate_limit import RateLimiter


def test_rate_limit():
    rl = RateLimiter(max_requests=2, window_sec=60)
    assert rl.is_allowed("k1")[0]
    assert rl.is_allowed("k1")[0]
    allowed, retry = rl.is_allowed("k1")
    assert not allowed and retry > 0
    rl.reset("k1")
    assert rl.is_allowed("k1")[0]

