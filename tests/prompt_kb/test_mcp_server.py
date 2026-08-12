"""Unit tests for the MCP server tools (no stdio, no real model)."""

import sqlite3
import urllib.error

import pytest

pytest.importorskip("numpy")  # search depends on the [rag] extra

from claude_tap.prompt_kb import mcp_server  # noqa: E402
from claude_tap.prompt_kb.embed import EmbedderUnavailable  # noqa: E402
from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending  # noqa: E402
from claude_tap.prompt_kb.store import KbStore  # noqa: E402
from tests.prompt_kb.fake_embedder import FakeEmbedder  # noqa: E402


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    """Isolated KB store + fake embedder injected into the module-level context."""
    store = KbStore(tmp_path / "kb.sqlite3")
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    monkeypatch.setattr(mcp_server, "_get_ctx", lambda: (store, embedder))
    monkeypatch.setattr(mcp_server.KbStore, "default", classmethod(lambda cls: store))
    return store, embedder


def _seed(store: KbStore) -> None:
    snap_id, _ = store.upsert_snapshot(
        content_hash="ha",
        client="codex",
        provider="openai",
        model="gpt-5",
        system_prompt="s",
        developer_prompt="",
        tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(snap_id, [("tool", "shell", "sandbox shell command runner")])
    store.upsert_message(
        session_id="s1",
        record_index=0,
        message_index=0,
        client="codex",
        model="gpt-5",
        timestamp="2026-08-01T00:00:00Z",
        content_hash="m0",
        text="how do I fix the race condition in the worker pool",
        seen_at="t",
    )
    index_pending(store, FakeEmbedder())


def test_kb_search_returns_chunks_section(ctx):
    store, _ = ctx
    _seed(store)
    result = mcp_server.kb_search("shell sandbox")
    assert result["chunks"][0]["client"] == "codex"
    assert result["chunks"][0]["model"] == "gpt-5"
    hit = result["chunks"][0]["hits"][0]
    assert hit["kind"] == "tool" and hit["title"] == "shell"
    assert hit["score"] > 0.5
    assert set(result) == {"chunks", "messages"}


def test_kb_search_returns_messages_section(ctx):
    store, _ = ctx
    _seed(store)
    result = mcp_server.kb_search("race condition lock")
    assert result["messages"][0]["session_id"] == "s1"
    hit = result["messages"][0]["hits"][0]
    assert "race condition" in hit["text"]
    assert hit["timestamp"] and hit["score"] > 0


def test_kb_search_messages_first_and_roles(monkeypatch):
    """kb_search 响应 messages 分区在前，hit 带 role，rel_delta 透传。"""
    calls = {}

    class FakeGroup:
        session_id = "s1"
        client = "claude"
        model = "k3"
        hits = [type("H", (), {"text": "t", "timestamp": "ts", "score": 0.9, "role": "assistant"})()]

    def fake_search_messages(store, embedder, query, **kw):
        calls.update(kw)
        return [FakeGroup()]

    monkeypatch.setattr("claude_tap.prompt_kb.mcp_server.search", lambda *a, **kw: [])
    monkeypatch.setattr("claude_tap.prompt_kb.mcp_server.search_messages", fake_search_messages)
    monkeypatch.setattr("claude_tap.prompt_kb.mcp_server.index_pending", lambda *a, **kw: None)
    monkeypatch.setattr("claude_tap.prompt_kb.mcp_server._get_ctx", lambda: (object(), object()))
    from claude_tap.prompt_kb.mcp_server import kb_search

    result = kb_search("q", rel_delta=0.1)
    assert list(result.keys())[0] == "messages"
    assert result["messages"][0]["hits"][0]["role"] == "assistant"
    assert calls["rel_delta"] == 0.1


def test_kb_search_indexes_pending_first(ctx):
    """New traces must be searchable without an explicit reindex."""
    store, _ = ctx
    _seed(store)
    # Arrives after the initial index: stays pending until kb_search runs.
    store.upsert_message(
        session_id="s2",
        record_index=1,
        message_index=0,
        client="codex",
        model="gpt-5",
        timestamp="2026-08-02T00:00:00Z",
        content_hash="m1",
        text="kubectl restart the deadlock pod",
        seen_at="t",
    )
    result = mcp_server.kb_search("deadlock pod")
    texts = [h["text"] for g in result["messages"] for h in g["hits"]]
    assert any("deadlock" in t for t in texts)


def test_kb_status(ctx):
    store, _ = ctx
    _seed(store)
    status = mcp_server.kb_status()
    assert status["embedder"] == "fake"
    for key in ("snapshots", "chunks", "pending", "failed", "indexed", "messages"):
        assert key in status
    assert status["snapshots"] == 1 and status["messages"] == 1


def test_kb_search_embedder_unavailable(monkeypatch):
    def _raise():
        raise EmbedderUnavailable("sentence-transformers is not installed")

    monkeypatch.setattr(mcp_server, "_get_ctx", _raise)
    result = mcp_server.kb_search("anything")
    assert "sentence-transformers" in result["error"]
    assert result["chunks"] == [] and result["messages"] == []


def test_kb_search_query_time_embedder_unavailable(ctx, monkeypatch):
    """An embedder failure at query time must honor the error-dict contract."""

    def _raise(*args, **kwargs):
        raise EmbedderUnavailable("numpy is not installed")

    monkeypatch.setattr(mcp_server, "search", _raise)
    result = mcp_server.kb_search("anything")
    assert "embedder unavailable" in result["error"]
    assert "numpy" in result["error"]
    assert result["chunks"] == [] and result["messages"] == []


def test_kb_search_query_time_network_error(ctx, monkeypatch):
    """An ApiEmbedder going offline at query time must not raise to the client."""

    def _raise(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(mcp_server, "search", _raise)
    result = mcp_server.kb_search("anything")
    assert "offline" in result["error"]
    assert result["chunks"] == [] and result["messages"] == []


def test_kb_status_operational_error(ctx, monkeypatch):
    """A locked DB on the read path must return an error dict, not raise."""

    def _locked(self):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(KbStore, "stats", _locked)
    result = mcp_server.kb_status()
    assert "locked" in result["error"]


def test_kb_search_reindex_required(ctx, monkeypatch):
    store, _ = ctx
    _seed(store)
    other = FakeEmbedder()
    other.name = "other"  # instance attribute shadows the class attribute
    monkeypatch.setattr(mcp_server, "_get_ctx", lambda: (store, other))
    result = mcp_server.kb_search("shell sandbox")
    assert "reindex" in result["error"]
    assert result["chunks"] == [] and result["messages"] == []


def test_kb_search_survives_index_lock(ctx, monkeypatch):
    """A locked DB (dashboard lazy indexer) must not fail the search."""
    store, _ = ctx
    _seed(store)

    def _locked(store, embedder, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(mcp_server, "index_pending", _locked)
    result = mcp_server.kb_search("shell sandbox")
    assert "error" not in result
    assert result["chunks"][0]["hits"][0]["title"] == "shell"


def test_main_without_mcp_extra(capsys, monkeypatch):
    monkeypatch.setattr(mcp_server, "FastMCP", None)
    assert mcp_server.main() == 2
    assert "claude-tap[mcp,rag]" in capsys.readouterr().err


def test_main_registers_tools_and_runs_stdio(monkeypatch):
    class _FakeFastMCP:
        instances = []

        def __init__(self, name):
            self.name = name
            self.tools = []
            self.ran = False
            _FakeFastMCP.instances.append(self)

        def tool(self):
            def decorate(fn):
                self.tools.append(fn.__name__)
                return fn

            return decorate

        def run(self):
            self.ran = True

    monkeypatch.setattr(mcp_server, "FastMCP", _FakeFastMCP)
    assert mcp_server.main() == 0
    server = _FakeFastMCP.instances[0]
    assert server.tools == ["kb_search", "kb_status"]
    assert server.ran


def test_cli_dispatch_mcp(monkeypatch):
    import sys

    import claude_tap.cli as cli

    monkeypatch.setattr(sys, "argv", ["claude-tap", "mcp"])
    monkeypatch.setattr(mcp_server, "main", lambda: 0)
    with pytest.raises(SystemExit) as exc_info:
        cli.main_entry()
    assert exc_info.value.code == 0


def test_cli_dispatch_mcp_help(capsys, monkeypatch):
    """`claude-tap mcp --help` must print usage instead of launching the server."""
    import sys

    import claude_tap.cli as cli

    def _forbidden():
        raise AssertionError("server must not launch for --help")

    monkeypatch.setattr(sys, "argv", ["claude-tap", "mcp", "--help"])
    monkeypatch.setattr(mcp_server, "main", _forbidden)
    with pytest.raises(SystemExit) as exc_info:
        cli.main_entry()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "Usage: claude-tap mcp" in out
    assert "claude-tap[mcp,rag]" in out
