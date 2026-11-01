import pytest
from fastapi.testclient import TestClient

from services.api.app import app
from services.api.core import backend_factory
from services.api.core.config import RuntimeConfig, normalize_backend_name
from services.api.core.engine_backend import EngineBackend
from services.api.core.example_backend import CannedBackend, HFExampleBackend
from services.api.core.mock_backend import MockBackend

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_default_mock_backend():
    response = client.post(
        "/v1/chat",
        json={"message": "বাংলায় একটি সংক্ষিপ্ত পরিচয় দাও।"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "kothagpt"
    assert body["message"]
    assert body["output"]


def test_chat_completions_default_mock_backend():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "kothagpt", "messages": [{"role": "user", "content": "হ্যালো"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]


def test_canned_backend(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "canned")
    backend = backend_factory.create()
    assert isinstance(backend, CannedBackend)
    output = backend.chat(_chat_request("আমি কোন মডেল ব্যবহার করছি?")).choices[0].message.content
    assert "ক্যানড" in output
    assert "আমি কোন মডেল" in output


def test_hf_backend_selection(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "hf")
    backend = backend_factory.create()
    assert isinstance(backend, HFExampleBackend)


def test_hf_backend_falls_back_to_canned_without_deps(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "hf")
    backend = backend_factory.create()
    output = backend.chat(_chat_request("পরীক্ষা")).choices[0].message.content
    # transformers/torch are not installed in the test env, so the shim
    # must degrade to the canned reply instead of raising.
    assert "ক্যানড" in output


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "does-not-exist")
    with pytest.raises(ValueError, match="does-not-exist"):
        RuntimeConfig.from_env()

    with pytest.raises(ValueError, match="does-not-exist"):
        backend_factory.create("does-not-exist")


def test_factory_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("KOTHAGPT_BACKEND", raising=False)
    assert isinstance(backend_factory.create(), MockBackend)


def test_backend_registration_requires_backend_contract():
    with pytest.raises(TypeError, match="Backend subclass"):
        backend_factory.register("custom", object)


def test_runtime_config_validates_backend_and_limits(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "mock")
    config = RuntimeConfig.from_env()
    assert config.backend == "mock"
    assert config.model_id == "kothagpt"
    assert config.device == "cpu"
    assert config.quant == "none"

    monkeypatch.setenv("KOTHAGPT_BACKEND", "real")
    real_cfg = RuntimeConfig.from_env()
    assert real_cfg.backend == "engine"

    monkeypatch.setenv("KOTHAGPT_MODEL_ID", "kothagpt-small")
    monkeypatch.setenv("KOTHAGPT_DEVICE", "cuda:0")
    monkeypatch.setenv("KOTHAGPT_QUANT", "bf16")
    tuned_cfg = RuntimeConfig.from_env()
    assert tuned_cfg.model_id == "kothagpt-small"
    assert tuned_cfg.device == "cuda:0"
    assert tuned_cfg.quant == "bf16"

    monkeypatch.setenv("KOTHAGPT_BACKEND", "Bad Backend")
    with pytest.raises(ValueError, match="KOTHAGPT_BACKEND"):
        RuntimeConfig.from_env()

    monkeypatch.setenv("KOTHAGPT_BACKEND", "mock")
    monkeypatch.setenv("KOTHAGPT_MAX_NEW_TOKENS", "0")
    with pytest.raises(ValueError, match="KOTHAGPT_MAX_NEW_TOKENS"):
        RuntimeConfig.from_env()

    monkeypatch.setenv("KOTHAGPT_MAX_NEW_TOKENS", "32")
    monkeypatch.setenv("KOTHAGPT_QUANT", "fp16")
    with pytest.raises(ValueError, match="KOTHAGPT_QUANT"):
        RuntimeConfig.from_env()


def test_backend_capabilities_are_explicit_and_registry_backed():
    caps = backend_factory.create("mock").get_capabilities()
    assert caps.chat is True
    assert caps.streaming is True
    assert caps.tools is True
    assert backend_factory.describe("mock")["mock"]["agents"] is True


def test_backend_registry_endpoint_exposes_capabilities():
    response = client.get("/v1/models/backends")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert any(item["name"] == "mock" for item in payload["data"])
    assert payload["data"][0]["capabilities"]["chat"] is True


def test_model_registry_reports_runtime_statuses():
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert any(item["id"] == "kothagpt" and item.get("status") == "ready" for item in payload["data"])
    assert backend_factory.create("mock").list_models()[0].status == "ready"


def test_real_backend_alias_registers_engine_backend(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "real")
    backend = backend_factory.create()
    assert isinstance(backend, EngineBackend)


def _chat_request(message: str):
    from services.api.api.schemas import ChatCompletionRequest

    return ChatCompletionRequest(messages=[{"role": "user", "content": message}])
