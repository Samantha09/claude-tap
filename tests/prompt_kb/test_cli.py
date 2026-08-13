import pytest

from claude_tap.prompt_kb.cli import kb_main
from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending
from claude_tap.prompt_kb.store import KbStore
from tests.prompt_kb.fake_embedder import FakeEmbedder


@pytest.fixture()
def seeded_kb(trace_db, monkeypatch):
    store = KbStore.default()
    snap_id, _ = store.upsert_snapshot(
        content_hash="h",
        client="codex",
        provider="openai",
        model="gpt-5",
        system_prompt="s",
        developer_prompt="",
        tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(snap_id, [("tool", "shell", "sandbox shell command runner")])
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    monkeypatch.setattr("claude_tap.prompt_kb.cli.create_embedder", lambda config: embedder)
    monkeypatch.setattr("claude_tap.prompt_kb.cli.create_reranker", lambda config: None)
    return store


def test_kb_search_prints_grouped_results(seeded_kb, capsys):
    assert kb_main(["search", "shell sandbox"]) == 0
    out = capsys.readouterr().out
    assert "codex" in out and "gpt-5" in out and "shell" in out


def test_kb_search_prints_reranked_line(seeded_kb, capsys):
    assert kb_main(["search", "shell sandbox"]) == 0
    assert "reranked: no" in capsys.readouterr().out


def test_kb_status_prints_counts(seeded_kb, capsys):
    assert kb_main(["status"]) == 0
    out = capsys.readouterr().out
    assert "indexed=1" in out


def test_kb_status_prints_reranker_line(seeded_kb, capsys):
    assert kb_main(["status"]) == 0
    assert "reranker=" in capsys.readouterr().out


def test_kb_reindex(seeded_kb, capsys):
    assert kb_main(["reindex"]) == 0
    out = capsys.readouterr().out
    assert "indexed=1" in out
    assert "messages_indexed=0" in out
    assert "messages_failed=0" in out


def test_kb_search_prints_message_section(trace_db, monkeypatch, capsys):
    store = KbStore.default()
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    store.upsert_message(
        session_id="sess-1",
        record_index=0,
        message_index=0,
        client="claude",
        model="k3",
        timestamp="2026-08-10T01:00:00Z",
        content_hash="h1",
        text="how to fix the race condition",
        seen_at="t",
    )
    index_pending(store, embedder)
    monkeypatch.setattr("claude_tap.prompt_kb.cli.create_embedder", lambda config: embedder)
    rc = kb_main(["search", "race condition"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "messages:" in out
    assert "sess-1" in out


def test_cli_search_prints_messages_first_with_roles(trace_db, monkeypatch, capsys):
    """messages 小节先于快照组打印，消息行带 [user]/[assistant] 前缀。"""
    store = KbStore.default()
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    snap_id, _ = store.upsert_snapshot(
        content_hash="h",
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
        session_id="sess-1",
        record_index=0,
        message_index=0,
        client="claude",
        model="k3",
        timestamp="2026-08-10T01:00:00Z",
        content_hash="h1",
        text="how to fix the race condition",
        seen_at="t",
    )
    index_pending(store, embedder)
    monkeypatch.setattr("claude_tap.prompt_kb.cli.create_embedder", lambda config: embedder)
    rc = kb_main(["search", "race condition shell"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "messages:" in out
    assert "[1]" in out
    assert out.index("messages:") < out.index("[1]")
    assert "[user]" in out


def test_kb_search_embedder_unavailable(trace_db, monkeypatch, capsys):
    from claude_tap.prompt_kb.embed import EmbedderUnavailable

    def _raise(config):
        raise EmbedderUnavailable("no model")

    monkeypatch.setattr("claude_tap.prompt_kb.cli.create_embedder", _raise)
    assert kb_main(["search", "x"]) == 2
    assert "no model" in capsys.readouterr().err
