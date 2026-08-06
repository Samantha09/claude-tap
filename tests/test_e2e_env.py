"""e2e subprocess dashboards must use ephemeral ports, never the fixed 19527."""

from __future__ import annotations

import asyncio
import socket

import aiohttp
import pytest

from claude_tap.live import LiveViewerServer
from claude_tap.shared_dashboard import DEFAULT_DASHBOARD_PORT
from tests.conftest import allocate_free_port, e2e_env, stop_dashboard_on_port


def test_e2e_env_assigns_ephemeral_dashboard_port(tmp_path):
    env1 = e2e_env({}, tmp_path)
    env2 = e2e_env({}, tmp_path)
    port1 = int(env1["CLOUDTAP_DASHBOARD_PORT"])
    port2 = int(env2["CLOUDTAP_DASHBOARD_PORT"])
    assert port1 > 0
    assert port2 > 0
    assert port1 != port2
    assert DEFAULT_DASHBOARD_PORT not in (port1, port2)


async def test_stop_dashboard_on_port_stops_real_server():
    port = allocate_free_port()
    server = LiveViewerServer(port=port, dashboard_mode=True)
    await server.start()

    async def _watch_shutdown() -> None:
        await server.wait_stopped()
        await server.stop()

    watcher = asyncio.create_task(_watch_shutdown())
    try:
        # to_thread keeps this test's loop free to serve the quit request;
        # real call sites are sync e2e tests where blocking is fine.
        await asyncio.to_thread(stop_dashboard_on_port, port)
        await asyncio.wait_for(watcher, timeout=25)
        with pytest.raises((aiohttp.ClientError, OSError)):
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/dashboard/health"):
                    pass
    finally:
        await server.stop()


def test_allocate_free_port_skips_already_allocated():
    port = allocate_free_port()
    # The port stays reserved, so a second allocation must not repeat it.
    assert allocate_free_port() != port
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
