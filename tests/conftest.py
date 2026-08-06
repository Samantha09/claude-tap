"""Pytest configuration and shared fixtures."""

import asyncio
import os
import shutil
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from claude_tap.cli_clients import _extend_no_proxy
from claude_tap.shared_dashboard import DEFAULT_DASHBOARD_PORT, stop_dashboard_service
from claude_tap.trace_store import get_trace_store, reset_trace_store

# Ports handed out to e2e subprocess dashboards this run. Tracked so the
# autouse cleanup fixture can stop every spawned dashboard afterwards and
# so allocations never repeat a port (or the fixed default dashboard port).
_e2e_dashboard_ports: set[int] = set()


def allocate_free_port() -> int:
    """Reserve an ephemeral loopback port for one e2e dashboard."""
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port != DEFAULT_DASHBOARD_PORT and port not in _e2e_dashboard_ports:
            _e2e_dashboard_ports.add(port)
            return port


def stop_dashboard_on_port(port: int) -> None:
    """Stop a dashboard service from sync test code (own thread + loop)."""
    result: list[bool] = []

    def _run() -> None:
        result.append(asyncio.run(stop_dashboard_service("127.0.0.1", port)))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=20)


def trace_db_path(trace_dir: str | Path) -> Path:
    return Path(trace_dir) / "claude-tap-test.sqlite3"


def e2e_env(env: dict[str, str], trace_dir: str | Path) -> dict[str, str]:
    updated = dict(env)
    updated["CLOUDTAP_DB"] = str(trace_db_path(trace_dir))
    updated["CLOUDTAP_DASHBOARD_PORT"] = str(allocate_free_port())
    _extend_no_proxy(updated, ("localhost", "127.0.0.1", "::1"))
    return updated


def read_trace_records(trace_dir: str | Path, *, session_index: int = -1) -> list[dict]:
    db_path = trace_db_path(trace_dir)
    reset_trace_store()
    os.environ["CLOUDTAP_DB"] = str(db_path)
    store = get_trace_store()
    rows = store.list_session_rows()
    if not rows:
        return []
    session_id = rows[session_index]["id"]
    return store.load_records(session_id)


def read_proxy_log(trace_dir: str | Path, *, session_index: int = -1) -> str:
    db_path = trace_db_path(trace_dir)
    reset_trace_store()
    os.environ["CLOUDTAP_DB"] = str(db_path)
    store = get_trace_store()
    rows = store.list_session_rows()
    if not rows:
        return ""
    session_id = rows[session_index]["id"]
    return store.export_log(session_id)


@pytest.fixture(autouse=True)
def stop_e2e_dashboards():
    """Stop every dashboard spawned by e2e subprocess runs (port-tracked)."""
    yield
    for port in list(_e2e_dashboard_ports):
        _e2e_dashboard_ports.discard(port)
        try:
            stop_dashboard_on_port(port)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def isolate_trace_store():
    """Reset the process-wide TraceStore singleton and CLOUDTAP_DB between tests."""
    saved_db = os.environ.get("CLOUDTAP_DB")
    os.environ.pop("CLOUDTAP_DB", None)
    reset_trace_store()
    yield
    reset_trace_store()
    if saved_db is None:
        os.environ.pop("CLOUDTAP_DB", None)
    else:
        os.environ["CLOUDTAP_DB"] = saved_db


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    """Provide an isolated SQLite trace database for each test."""
    db_path = tmp_path / "test-traces.sqlite3"
    monkeypatch.setenv("CLOUDTAP_DB", str(db_path))
    reset_trace_store()
    yield db_path
    reset_trace_store()


@pytest.fixture
def temp_trace_dir():
    """Create a temporary directory for trace output."""
    trace_dir = tempfile.mkdtemp(prefix="claude_tap_test_")
    yield trace_dir
    shutil.rmtree(trace_dir, ignore_errors=True)


@pytest.fixture
def temp_bin_dir():
    """Create a temporary directory for fake binaries."""
    bin_dir = tempfile.mkdtemp(prefix="claude_tap_bin_")
    yield bin_dir
    shutil.rmtree(bin_dir, ignore_errors=True)


@pytest.fixture
def project_dir():
    """Return the project root directory."""
    return Path(__file__).parent.parent
