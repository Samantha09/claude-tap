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
    assert config.embedder == "local"  # env wins
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
        create_embedder(
            KbConfig(
                embedder="api",
                api_base="https://x.example/v1",
                api_model="m",
                api_key_env="KB_TEST_KEY",
            )
        )


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
        return sum(p * q for p, q in zip(x, y)) / (math.sqrt(sum(p * p for p in x)) * math.sqrt(sum(q * q for q in y)))

    assert cos(a, b) > cos(a, c)
    assert len(a) == 16


def _fake_st_module(recorder: list) -> "object":
    import types

    fake = types.ModuleType("sentence_transformers")

    class _FakeST:
        def __init__(self, model_name):
            self.model_name = model_name

        def get_embedding_dimension(self):
            return 4

        def encode(self, texts, normalize_embeddings=True):
            recorder.extend(texts)
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    fake.SentenceTransformer = _FakeST
    return fake


def test_local_embedder_applies_e5_prefixes(monkeypatch):
    """E5-family models require 'query: '/'passage: ' prefixes; without them
    similarity scores compress into a narrow band and junk scores high."""
    import sys

    recorder: list = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st_module(recorder))
    emb = embed_mod.LocalEmbedder("some-model")
    emb.embed(["doc text"])
    emb.embed_query(["user query"])
    assert recorder == ["passage: doc text", "query: user query"]


def test_local_embedder_name_marks_prefix_space(monkeypatch):
    """Prefixing changes the vector space; the embedder name must differ from
    pre-prefix indexes so kb_meta triggers ReindexRequired instead of mixing."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st_module([]))
    emb = embed_mod.LocalEmbedder("some-model")
    assert emb.name != "local:some-model"
    assert emb.name.startswith("local:some-model")


def test_canonical_model_id_passes_plain_ids_through():
    assert embed_mod.canonical_model_id("intfloat/multilingual-e5-small") == "intfloat/multilingual-e5-small"
    assert embed_mod.canonical_model_id("some-model") == "some-model"


def test_canonical_model_id_resolves_modelscope_hub_paths():
    assert (
        embed_mod.canonical_model_id("/Users/san/.cache/modelscope/hub/models/intfloat/multilingual-e5-small")
        == "intfloat/multilingual-e5-small"
    )
    # legacy layout without the models/ segment
    assert (
        embed_mod.canonical_model_id("/root/.cache/modelscope/hub/BAAI/bge-reranker-base") == "BAAI/bge-reranker-base"
    )
    # trailing slash tolerated
    assert (
        embed_mod.canonical_model_id("/Users/san/.cache/modelscope/hub/models/BAAI/bge-reranker-base/")
        == "BAAI/bge-reranker-base"
    )


def test_canonical_model_id_resolves_hf_cache_paths():
    assert (
        embed_mod.canonical_model_id(
            "/home/u/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots/ab12cd34"
        )
        == "intfloat/multilingual-e5-small"
    )


def test_canonical_model_id_keeps_unknown_absolute_paths():
    # A bare directory is the identity: no org/name structure to recover.
    assert embed_mod.canonical_model_id("/opt/models/my-fine-tune") == "/opt/models/my-fine-tune"


def test_local_embedder_name_uses_canonical_id_not_path(monkeypatch):
    """A modelscope/hf cache path and the plain model id must produce the SAME
    embedder name, so moving the cache never forces a reindex."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st_module([]))
    by_path = embed_mod.LocalEmbedder("/Users/san/.cache/modelscope/hub/models/intfloat/multilingual-e5-small")
    by_id = embed_mod.LocalEmbedder("intfloat/multilingual-e5-small")
    assert by_path.name == by_id.name
    assert "modelscope" not in by_path.name


def test_api_embedder_embed_query_has_no_prefix(monkeypatch):
    """API embedders have no prefix convention: embed_query delegates to embed."""
    emb = embed_mod.ApiEmbedder(api_base="https://x.example", api_model="m", api_key="k")
    calls = []
    monkeypatch.setattr(emb, "embed", lambda texts: calls.append(texts) or [[0.1]])
    assert emb.embed_query(["a"]) == [[0.1]]
    assert calls == [["a"]]


def test_load_config_prefix_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_TAP_KB_QUERY_PREFIX", "q:")
    config = load_config(tmp_path / "missing.toml")
    assert config.query_prefix == "q:"
    assert config.passage_prefix == "passage: "
