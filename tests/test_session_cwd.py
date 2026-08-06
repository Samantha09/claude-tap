"""Sessions record the process working directory (schema v5)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from claude_tap.trace import create_trace_writer
from claude_tap.trace_store import SCHEMA_VERSION, TraceStore


def _session_row(db_path: Path, session_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    finally:
        conn.close()


def test_create_session_records_explicit_cwd(tmp_path):
    db_path = tmp_path / "traces.sqlite3"
    store = TraceStore(db_path)
    session_id = store.create_session(client="claude", cwd="/proj/demo")
    assert _session_row(db_path, session_id)["cwd"] == "/proj/demo"


def test_create_session_defaults_cwd_to_empty(tmp_path):
    db_path = tmp_path / "traces.sqlite3"
    store = TraceStore(db_path)
    session_id = store.create_session(client="claude")
    assert _session_row(db_path, session_id)["cwd"] == ""


def test_trace_writer_captures_process_cwd(tmp_path, monkeypatch):
    db_path = tmp_path / "traces.sqlite3"
    store = TraceStore(db_path)
    monkeypatch.chdir("/tmp")
    writer = create_trace_writer(store=store, client="claude", proxy_mode="reverse", metadata={})
    assert _session_row(db_path, writer.session_id)["cwd"] == os.getcwd()


def test_trace_writer_honors_explicit_cwd(tmp_path):
    db_path = tmp_path / "traces.sqlite3"
    store = TraceStore(db_path)
    writer = create_trace_writer(store=store, client="claude", proxy_mode="reverse", metadata={}, cwd="/proj/override")
    assert _session_row(db_path, writer.session_id)["cwd"] == "/proj/override"


def test_v4_database_migrates_to_v5_keeping_legacy_rows(tmp_path):
    db_path = tmp_path / "traces.sqlite3"
    store = TraceStore(db_path)
    session_id = store.create_session(client="claude", cwd="/proj/demo")
    del store

    # Downgrade to a v4-shaped database: sessions table without the cwd column.
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE sessions_v4 (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            date_key TEXT NOT NULL,
            client TEXT NOT NULL DEFAULT '',
            proxy_mode TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            record_count INTEGER NOT NULL DEFAULT 0,
            summary_json TEXT,
            legacy_source_key TEXT NOT NULL DEFAULT '',
            legacy_rel_path TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO sessions_v4 (
            id, started_at, updated_at, date_key, client, proxy_mode, status, record_count,
            summary_json, legacy_source_key, legacy_rel_path
        )
        SELECT id, started_at, updated_at, date_key, client, proxy_mode, status, record_count,
               summary_json, legacy_source_key, legacy_rel_path
        FROM sessions
        """
    )
    conn.execute("DROP TABLE sessions")
    conn.execute("ALTER TABLE sessions_v4 RENAME TO sessions")
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()

    migrated = TraceStore(db_path)
    migrated.list_session_rows()  # triggers lazy schema migration
    row = _session_row(db_path, session_id)
    assert row["cwd"] == ""
    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert version == SCHEMA_VERSION


def test_notes_era_v6_database_is_accepted(tmp_path):
    """Databases stamped v6 by the abandoned notes app stay readable.

    The notes-era schema added cwd plus orphaned tables (tags, notes,
    summaries). Those tables are no longer managed, but the store must
    accept the database instead of rejecting it as "unsupported".
    """
    db_path = tmp_path / "traces.sqlite3"
    store = TraceStore(db_path)
    session_id = store.create_session(client="claude", cwd="/proj/demo")
    store.append_record(
        session_id,
        {
            "timestamp": "2026-08-05T09:00:00+00:00",
            "turn": 1,
            "request": {"method": "POST", "path": "/v1/messages", "body": {"model": "opus"}},
            "response": {"status": 200, "body": {"usage": {"input_tokens": 10, "output_tokens": 5}}},
        },
    )
    del store

    # Simulate the notes-era stamp: extra orphaned tables plus user_version 6.
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, body TEXT)")
    conn.execute("CREATE TABLE tags (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("PRAGMA user_version = 6")
    conn.commit()
    conn.close()

    reopened = TraceStore(db_path)
    rows = reopened.list_session_rows()
    assert [row["id"] for row in rows] == [session_id]
    assert rows[0]["cwd"] == "/proj/demo"
