from ml.inference.version import VersionManager


def test_version_flow():
    vm = VersionManager()
    v1 = vm.register_version("kothagpt", "ml/pretrain/artifacts/v1")
    v2 = vm.register_version("kothagpt", "ml/pretrain/artifacts/v2")
    assert v1.version != v2.version
    assert vm.get_stable("kothagpt").version == v1.version
    vm.promote_stable("kothagpt", v2.version)
    assert vm.get_stable("kothagpt").version == v2.version
    vm.rollback("kothagpt")
    assert vm.get_stable("kothagpt").version == v1.version

