"""SQLite storage for the prompt knowledge base.

The KB database is derived data: it can always be rebuilt from the trace
store. It lives next to the trace database (CLOUDTAP_DB) so tests that
redirect the trace DB automatically get an isolated KB.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from claude_tap.trace_store import resolve_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_snapshots (
  id INTEGER PRIMARY KEY,
  content_hash TEXT NOT NULL,
  client TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  system_prompt TEXT,
  developer_prompt TEXT,
  tools_json TEXT,
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  session_count INTEGER NOT NULL DEFAULT 1,
  UNIQUE(client, model, content_hash)
);
CREATE TABLE IF NOT EXISTS kb_chunks (
  id INTEGER PRIMARY KEY,
  snapshot_id INTEGER NOT NULL REFERENCES kb_snapshots(id),
  kind TEXT NOT NULL,
  title TEXT,
  text TEXT NOT NULL,
  embedding BLOB,
  index_state TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_state ON kb_chunks(index_state);
CREATE TABLE IF NOT EXISTS kb_sources (
  session_id TEXT PRIMARY KEY,
  snapshot_id INTEGER REFERENCES kb_snapshots(id),
  processed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kb_meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""

def default_db_path() -> Path:
    # Reuse the trace DB resolution (CLOUDTAP_DB / XDG_DATA_HOME) so the KB
    # always lands next to the trace database, including in tests.
    return resolve_db_path().with_name("prompt_kb.sqlite3")


class KbStore:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @classmethod
    def default(cls) -> "KbStore":
        return cls(default_db_path())

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_snapshot(self, *, content_hash: str, client: str, provider: str,
                        model: str, system_prompt: str, developer_prompt: str,
                        tools_json: str, seen_at: str) -> tuple[int, bool]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, session_count FROM kb_snapshots WHERE client=? AND model=? AND content_hash=?",
                (client, model, content_hash),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE kb_snapshots SET session_count=session_count+1, last_seen=? WHERE id=?",
                    (seen_at, row["id"]),
                )
                return int(row["id"]), False
            cur = conn.execute(
                """INSERT INTO kb_snapshots
                   (content_hash, client, provider, model, system_prompt, developer_prompt,
                    tools_json, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (content_hash, client, provider, model, system_prompt,
                 developer_prompt, tools_json, seen_at, seen_at),
            )
            return int(cur.lastrowid), True

    def replace_chunks(self, snapshot_id: int, chunks: list[tuple[str, str, str]]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM kb_chunks WHERE snapshot_id=?", (snapshot_id,))
            conn.executemany(
                "INSERT INTO kb_chunks (snapshot_id, kind, title, text) VALUES (?, ?, ?, ?)",
                [(snapshot_id, kind, title, text) for kind, title, text in chunks],
            )

    def is_source_processed(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM kb_sources WHERE session_id=?", (session_id,)
            ).fetchone()
            return row is not None

    def record_source(self, session_id: str, snapshot_id: int | None, processed_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kb_sources (session_id, snapshot_id, processed_at) VALUES (?, ?, ?)",
                (session_id, snapshot_id, processed_at),
            )

    def pending_chunks(self, limit: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, text FROM kb_chunks WHERE index_state='pending' ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()

    def mark_chunk_indexed(self, chunk_id: int, embedding: bytes) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE kb_chunks SET embedding=?, index_state='indexed' WHERE id=?",
                (embedding, chunk_id),
            )

    def mark_chunk_failed(self, chunk_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE kb_chunks SET index_state='failed' WHERE id=?", (chunk_id,)
            )

    def indexed_chunks(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """SELECT c.id, c.snapshot_id, c.kind, c.title, c.text, c.embedding,
                          s.client, s.model, s.first_seen, s.last_seen, s.session_count
                   FROM kb_chunks c JOIN kb_snapshots s ON s.id = c.snapshot_id
                   WHERE c.index_state='indexed'"""
            ).fetchall()

    def reset_embeddings(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE kb_chunks SET embedding=NULL, index_state='pending'"
            )
            return cur.rowcount

    def get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kb_meta WHERE key=?", (key,)).fetchone()
            return str(row["value"]) if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kb_meta (key, value) VALUES (?, ?)", (key, value)
            )

    def get_snapshot(self, snapshot_id: int) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM kb_snapshots WHERE id=?", (snapshot_id,)
            ).fetchone()

    def timeline(self, client: str, model: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """SELECT id, content_hash, first_seen, last_seen, session_count
                   FROM kb_snapshots WHERE client=? AND model=? ORDER BY first_seen ASC, id ASC""",
                (client, model),
            ).fetchall()

    def stats(self) -> dict:
        with self._connect() as conn:
            snapshots = conn.execute("SELECT COUNT(*) c FROM kb_snapshots").fetchone()["c"]
            chunks = conn.execute("SELECT COUNT(*) c FROM kb_chunks").fetchone()["c"]
            by_state = {
                row["index_state"]: row["c"]
                for row in conn.execute(
                    "SELECT index_state, COUNT(*) c FROM kb_chunks GROUP BY index_state"
                )
            }
            return {
                "snapshots": int(snapshots),
                "chunks": int(chunks),
                "pending": int(by_state.get("pending", 0)),
                "failed": int(by_state.get("failed", 0)),
                "indexed": int(by_state.get("indexed", 0)),
            }
