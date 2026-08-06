import aiohttp
import pytest

from claude_tap.live import LiveViewerServer
from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending
from claude_tap.prompt_kb.store import KbStore
from tests.prompt_kb.fake_embedder import FakeEmbedder

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def seeded_kb(trace_db, monkeypatch):
    store = KbStore.default()
    snap_id, _ = store.upsert_snapshot(
        content_hash="h", client="codex", provider="openai", model="gpt-5",
        system_prompt="s", developer_prompt="", tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(snap_id, [("tool", "shell", "sandbox shell command runner")])
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    monkeypatch.setattr("claude_tap.live.create_embedder", lambda config: embedder)
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
