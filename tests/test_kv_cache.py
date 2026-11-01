import torch

from ml.inference.kv_cache import KVCache


def test_kv_cache():
    c = KVCache(num_layers=2, n_heads=2, head_dim=4, max_seq=8)
    k = torch.randn(1,2,3,4)
    v = torch.randn(1,2,3,4)
    c.update(0, k, v)
    assert c.seq_len(0) == 3
    k2 = torch.randn(1,2,2,4)
    v2 = torch.randn(1,2,2,4)
    c.update(0, k2, v2)
    assert c.seq_len(0) == 5
    c.clear()
    assert c.seq_len(0) == 0
