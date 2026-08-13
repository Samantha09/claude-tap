"""tokenize.segment(): jieba cut_for_search with graceful degradation."""

from __future__ import annotations

import builtins
import warnings

import pytest

from claude_tap.prompt_kb import tokenize


@pytest.fixture(autouse=True)
def _reset_jieba_cache(monkeypatch):
    monkeypatch.setattr(tokenize, "_jieba", None)
    monkeypatch.setattr(tokenize, "_jieba_failed", False)


def test_segment_exact_verified_output():
    assert tokenize.segment("取消定时任务cron") == "取消 定时 任务 cron"


def test_segment_mixed_keeps_english_tokens():
    tokens = tokenize.segment("用 CronDelete 取消定时任务").split()
    assert "CronDelete" in tokens
    assert "定时" in tokens


def test_segment_degrades_without_jieba(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jieba":
            raise ImportError("no jieba")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.warns(UserWarning, match="jieba"):
        assert tokenize.segment("取消定时任务") == "取消定时任务"
    # Second call: degradation is permanent for the process, no repeated warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tokenize.segment("再次调用") == "再次调用"


def test_module_top_does_not_import_jieba():
    source = open(__import__("claude_tap.prompt_kb.tokenize", fromlist=["x"]).__file__).read()
    assert "\nimport jieba" not in source
    assert "\nimport jieba." not in source
