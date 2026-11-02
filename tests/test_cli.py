import subprocess
import sys

import pytest


def run_cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "kothagpt_cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.fixture(scope="module")
def env(server):
    from tests.conftest import API_TEST_TOKEN

    return {"KOTHAGPT_API_URL": server, "KOTHAGPT_API_KEY": API_TEST_TOKEN}


def test_models(env):
    result = run_cli("models", env=env)
    assert result.returncode == 0
    assert "kothagpt" in result.stdout


def test_chat(env):
    result = run_cli("chat", "হ্যালো", env=env)
    assert result.returncode == 0
    assert result.stdout.strip()


def test_chat_stream(env):
    result = run_cli("chat", "--stream", "হ্যালো", env=env)
    assert result.returncode == 0
    assert result.stdout.strip()


def test_embed(env):
    result = run_cli("embed", "বাংলা", env=env)
    assert result.returncode == 0
    import json

    assert len(json.loads(result.stdout)) == 256


def test_tools(env):
    result = run_cli("tools", "list", env=env)
    assert result.returncode == 0
    assert "calculator" in result.stdout

    result = run_cli("tools", "invoke", "calculator", "--arg", "expression=2+3", env=env)
    assert result.returncode == 0
    assert '"value": 5' in result.stdout


def test_agents(env):
    result = run_cli("agents", "create", "--name", "cli-test", env=env)
    assert result.returncode == 0
    import json

    agent = json.loads(result.stdout)
    result = run_cli("agents", "run", agent["id"], "হাই", env=env)
    assert result.returncode == 0
    assert result.stdout.strip()
