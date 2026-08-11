import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def test_rag_extra_declared():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "rag" in extras
    assert any(dep.startswith("sentence-transformers") for dep in extras["rag"])
    assert any(dep.startswith("numpy") for dep in extras["rag"])


def test_prompt_kb_subpackage_distributed():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    packages = data["tool"]["setuptools"]["packages"]
    assert "claude_tap.prompt_kb" in packages


def test_public_api_reexports():
    from claude_tap.prompt_kb import KbStore, index_pending, rebuild_index, run_index_loop, search

    assert callable(search) and callable(index_pending)
    assert callable(rebuild_index) and callable(run_index_loop)
    assert KbStore is not None


def test_mcp_extra_declared():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "mcp" in extras
    assert any(dep.startswith("mcp") for dep in extras["mcp"])
