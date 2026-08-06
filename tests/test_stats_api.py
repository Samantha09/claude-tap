"""GET /api/stats on the shared dashboard server."""

from __future__ import annotations

from pathlib import Path

import aiohttp
import pytest

from claude_tap.live import LiveViewerServer
from claude_tap.trace_store import get_trace_store, reset_trace_store


@pytest.fixture
def stats_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLOUDTAP_DB", str(tmp_path / "stats.sqlite3"))
    reset_trace_store()
    store = get_trace_store()
    store.create_session(client="claude", cwd="/proj/x")
    return store


async def test_api_stats_returns_aggregates(stats_server):
    server = LiveViewerServer(port=0, dashboard_mode=True)
    port = await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/api/stats") as resp:
                assert resp.status == 200
                payload = await resp.json()
                assert payload["totals"]["sessions"] == 1
                assert payload["by_project"][0]["cwd"] == "/proj/x"
                assert payload["by_client"][0]["client"] == "claude"
    finally:
        await server.stop()


async def test_api_stats_rejects_invalid_date(stats_server):
    server = LiveViewerServer(port=0, dashboard_mode=True)
    port = await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/api/stats?from=2026-13-99") as resp:
                assert resp.status == 400
                assert "日期格式无效" in await resp.text()
            async with session.get(f"http://127.0.0.1:{port}/api/stats?from=2026-08-05&to=2026-08-01") as resp:
                assert resp.status == 400
                assert "起止颠倒" in await resp.text()
    finally:
        await server.stop()


async def test_api_stats_filters_by_date_range(stats_server):
    server = LiveViewerServer(port=0, dashboard_mode=True)
    port = await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/api/stats?from=1999-01-01&to=1999-01-02") as resp:
                assert resp.status == 200
                payload = await resp.json()
                assert payload["totals"]["sessions"] == 0
    finally:
        await server.stop()
