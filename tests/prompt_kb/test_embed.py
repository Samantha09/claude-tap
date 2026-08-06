# tests/prompt_kb/test_embed.py
import math

import pytest

from claude_tap.prompt_kb import embed as embed_mod
from claude_tap.prompt_kb.embed import (
    EmbedderUnavailable,
    KbConfig,
    create_embedder,
    load_config,
    vectors_to_blob,
)
from tests.prompt_kb.fake_embedder import FakeEmbedder


def test_load_config_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_TAP_KB_EMBEDDER", raising=False)
    config = load_config(tmp_path / "missing.toml")
    assert config.embedder == "local"
    assert config.local_model == "intfloat/multilingual-e5-small"


def test_load_config_toml_and_env_override(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[prompt_kb]\nembedder = "api"\napi_base = "https://x.example/v1"\napi_model = "m"\n')
    monkeypatch.setenv("CLAUDE_TAP_KB_EMBEDDER", "local")
    config = load_config(path)
    assert config.embedder == "local"          # env wins
    assert config.api_base == "https://x.example/v1"


def test_create_embedder_local_without_dependency(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
    with pytest.raises(EmbedderUnavailable):
        create_embedder(KbConfig(embedder="local"))


def test_local_embedder_wraps_model_load_errors(monkeypatch):
    """Model download/load failures (network, TLS, disk) must surface as
    EmbedderUnavailable, not leak as OSError 500s."""
    import sys
    import types

    fake = types.ModuleType("sentence_transformers")

    def _boom(model_name):
        raise OSError("TLS handshake failed")

    fake.SentenceTransformer = _boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    with pytest.raises(EmbedderUnavailable):
        embed_mod.LocalEmbedder("some-model")


def test_create_embedder_api_requires_key(monkeypatch):
    monkeypatch.delenv("KB_TEST_KEY", raising=False)
    with pytest.raises(EmbedderUnavailable):
        create_embedder(KbConfig(
            embedder="api", api_base="https://x.example/v1",
            api_model="m", api_key_env="KB_TEST_KEY",
        ))


def test_vectors_to_blob_roundtrip():
    blobs = vectors_to_blob([[1.0, 0.5], [0.0, 2.0]])
    import array
    values = array.array("f")
    values.frombytes(blobs[0])
    assert list(values) == [1.0, 0.5]
    assert len(blobs[0]) == 8  # 2 * float32


def test_fake_embedder_semantic():
    emb = FakeEmbedder()
    a = emb.embed(["sandbox shell command"])[0]
    b = emb.embed(["shell command runner"])[0]
    c = emb.embed(["totally unrelated topic"])[0]
    def cos(x, y):
        return sum(p * q for p, q in zip(x, y)) / (
            math.sqrt(sum(p * p for p in x)) * math.sqrt(sum(q * q for q in y)))
    assert cos(a, b) > cos(a, c)
    assert len(a) == 16
