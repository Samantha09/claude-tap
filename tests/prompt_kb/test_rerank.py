"""LocalReranker sigmoid scoring, degradation contract, and config wiring."""

from __future__ import annotations

import builtins
import math
from pathlib import Path

import pytest

from claude_tap.prompt_kb.embed import DEFAULT_RERANKER_MODEL, KbConfig, load_config
from claude_tap.prompt_kb.rerank import (
    LocalReranker,
    RerankerUnavailable,
    create_reranker,
    reranker_status,
)
from tests.prompt_kb.fake_reranker import FakeReranker


class _StubModel:
    def __init__(self, logits):
        self._logits = logits

    def predict(self, pairs):
        assert all(len(pair) == 2 for pair in pairs)
        return self._logits[: len(pairs)]


def _stub_reranker(logits):
    reranker = LocalReranker.__new__(LocalReranker)
    reranker._model = _StubModel(logits)
    return reranker


def test_rerank_sigmoid_normalizes():
    scores = _stub_reranker([0.0, 2.0, -2.0]).rerank("q", ["a", "b", "c"])
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert scores[2] == pytest.approx(1.0 / (1.0 + math.exp(2.0)))
    assert all(0.0 < score < 1.0 for score in scores)


def test_rerank_empty_texts():
    assert _stub_reranker([]).rerank("q", []) == []


def test_load_failure_raises_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("no sentence-transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RerankerUnavailable, match="sentence-transformers"):
        LocalReranker()


def test_create_reranker_off_returns_none():
    assert create_reranker(KbConfig(reranker="off")) is None


def test_reranker_status_states(monkeypatch):
    assert reranker_status(KbConfig(reranker="off")) == "off"
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert reranker_status(KbConfig()) == "unavailable"


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("CLAUDE_TAP_KB_RERANKER", "off")
    monkeypatch.setenv("CLAUDE_TAP_KB_RERANKER_MODEL", "org/custom-reranker")
    config = load_config(path=Path("/nonexistent-kb-config.toml"))
    assert config.reranker == "off"
    assert config.reranker_model == "org/custom-reranker"


def test_config_defaults():
    config = load_config(path=Path("/nonexistent-kb-config.toml"))
    assert config.reranker == "on"
    assert config.reranker_model == DEFAULT_RERANKER_MODEL == "BAAI/bge-reranker-base"


def test_fake_reranker_deterministic_contract():
    scores = FakeReranker().rerank("alpha beta", ["alpha beta gamma", "nothing here", "alpha"])
    assert scores == [1.0, 0.0, 0.5]
