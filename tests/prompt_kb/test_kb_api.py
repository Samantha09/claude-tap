import aiohttp
import pytest

from claude_tap.live import LiveViewerServer
from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending
from claude_tap.prompt_kb.store import KbStore
from claude_tap.trace_store import get_trace_store
from tests.prompt_kb.fake_embedder import FakeEmbedder

pytestmark = pytest.mark.asyncio


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
    monkeypatch.setattr("claude_tap.live.create_embedder", lambda config: embedder)
    monkeypatch.setattr("claude_tap.live.create_reranker", lambda config: None)
    return embedder


async def _get_json(port, path):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{port}{path}") as resp:
            return resp.status, await resp.json()


async def test_kb_search_route(trace_db, seeded_kb, tmp_path):
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        status, payload = await _get_json(port, "/api/kb/search?q=shell+sandbox")
        assert status == 200
        assert payload["results"][0]["client"] == "codex"
        assert payload["results"][0]["hits"][0]["title"] == "shell"
        assert payload["reranked"] is False
    finally:
        await server.stop()


async def test_kb_search_empty_query_has_full_shape(trace_db, seeded_kb, tmp_path):
    """Empty query must return the same keys as a real search (results/messages/reranked)."""
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        status, payload = await _get_json(port, "/api/kb/search?q=")
        assert status == 200
        assert payload["results"] == []
        assert payload["messages"] == []
        assert payload["reranked"] is False
    finally:
        await server.stop()


async def test_kb_status_route(trace_db, seeded_kb, tmp_path):
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        status, payload = await _get_json(port, "/api/kb/status")
        assert status == 200
        assert payload["available"] is True
        assert payload["stats"]["indexed"] == 1
    finally:
        await server.stop()


async def test_kb_search_unavailable_returns_501(trace_db, tmp_path, monkeypatch):
    from claude_tap.prompt_kb.embed import EmbedderUnavailable

    def _raise(config):
        raise EmbedderUnavailable("pip install 'claude-tap[rag]'")

    monkeypatch.setattr("claude_tap.live.create_embedder", _raise)
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        status, payload = await _get_json(port, "/api/kb/search?q=x")
        assert status == 501
        assert payload["error"] == "rag_extra_missing"
    finally:
        await server.stop()


async def test_kb_timeline_route(trace_db, seeded_kb, tmp_path):
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        status, payload = await _get_json(port, "/api/kb/timeline?client=codex&model=gpt-5")
        assert status == 200
        assert len(payload["versions"]) == 1
        assert payload["versions"][0]["first_seen"] == "2026-08-01T00:00:00Z"
    finally:
        await server.stop()


@pytest.fixture()
def seeded_kb_messages(trace_db, monkeypatch):
    store = KbStore.default()
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    store.upsert_message(
        session_id="sess-abc",
        record_index=0,
        message_index=0,
        client="claude",
        model="k3",
        timestamp="2026-08-10T01:00:00Z",
        content_hash="h1",
        text="how to fix the race condition",
        seen_at="2026-08-10T01:00:00Z",
    )
    index_pending(store, embedder)
    monkeypatch.setattr("claude_tap.live.create_embedder", lambda config: embedder)
    monkeypatch.setattr("claude_tap.live.create_reranker", lambda config: None)
    return embedder


async def test_kb_search_includes_messages(trace_db, seeded_kb_messages, tmp_path):
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        status, payload = await _get_json(port, "/api/kb/search?q=race+condition")
        assert status == 200
        assert "messages" in payload
        assert payload["messages"][0]["session_id"] == "sess-abc"
        assert payload["messages"][0]["hits"][0]["text"]
        assert all("role" in hit for g in payload["messages"] for hit in g["hits"])
        assert payload["results"] == []  # prompt partition still present
    finally:
        await server.stop()


async def test_delete_session_without_kb_never_creates_empty_db(trace_db, tmp_path):
    """Never-RAG users must not get an empty prompt_kb.sqlite3 from cascade delete.

    Non-dashboard mode: the dashboard's background indexer intentionally owns
    a KbStore; this test isolates the cascade-delete path.
    """
    from claude_tap.prompt_kb.store import default_db_path

    kb_path = default_db_path()
    assert not kb_path.exists()
    store = get_trace_store()
    session_id = store.create_session(client="claude", proxy_mode="reverse")
    store.finalize_session(session_id)

    server = LiveViewerServer(port=0)
    port = await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"http://127.0.0.1:{port}/api/sessions/{session_id}") as resp:
                assert resp.status == 200
        assert not kb_path.exists()
    finally:
        await server.stop()


async def test_delete_session_cascades_kb_messages(trace_db, seeded_kb_messages, tmp_path):
    store = KbStore.default()
    assert store.stats()["messages"] == 1
    server = LiveViewerServer(port=0, migrate_from=tmp_path, dashboard_mode=True)
    port = await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"http://127.0.0.1:{port}/api/sessions/sess-abc") as resp:
                assert resp.status in (200, 404)  # 404 if trace session absent; cascade must still run
        assert store.stats()["messages"] == 0
    finally:
        await server.stop()
