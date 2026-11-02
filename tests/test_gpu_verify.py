from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml.gpu_verify import inspect_environment, require_cuda


class FakeCuda:
    def is_available(self):
        return True

    def device_count(self):
        return 2

    def get_device_name(self, index):
        return f"Fake GPU {index}"

    def is_bf16_supported(self):
        return True


class FakeDistributed:
    def is_nccl_available(self):
        return True

    def is_gloo_available(self):
        return True


@pytest.fixture
def fake_torch():
    return SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda="12.4"),
        cuda=FakeCuda(),
        distributed=FakeDistributed(),
    )


def test_inspect_environment_reports_cuda_capabilities(fake_torch):
    summary = inspect_environment(fake_torch)
    assert summary["cuda_available"] is True
    assert summary["device_count"] == 2
    assert summary["devices"] == ["Fake GPU 0", "Fake GPU 1"]
    assert summary["bf16_supported"] is True
    assert summary["distributed_backends"] == {"nccl": True, "gloo": True}


def test_require_cuda_fails_actionably_without_cuda():
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        require_cuda({"cuda_available": False, "distributed_backends": {}})


def test_require_cuda_requires_nccl():
    with pytest.raises(RuntimeError, match="NCCL is unavailable"):
        require_cuda({"cuda_available": True, "distributed_backends": {"nccl": False}})
