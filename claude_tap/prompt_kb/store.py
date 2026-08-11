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
  index_state TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_state ON kb_chunks(index_state);
CREATE TABLE IF NOT EXISTS kb_sources (
  session_id TEXT PRIMARY KEY,
  snapshot_id INTEGER REFERENCES kb_snapshots(id),
  processed_at TEXT NOT NULL,
  messages_done INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kb_meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
CREATE TABLE IF NOT EXISTS kb_messages (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  record_index INTEGER NOT NULL,
  message_index INTEGER NOT NULL,
  client TEXT NOT NULL,
  model TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  text TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  embedding BLOB,
  index_state TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kb_messages_state ON kb_messages(index_state);
CREATE INDEX IF NOT EXISTS idx_kb_messages_session ON kb_messages(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_messages_dedup ON kb_messages(content_hash, client);
"""


def default_db_path() -> Path:
    # Reuse the trace DB resolution (CLOUDTAP_DB / XDG_DATA_HOME) so the KB
    # always lands next to the trace database, including in tests.
    return resolve_db_path().with_name("prompt_kb.sqlite3")


class KbStore:
    MAX_ATTEMPTS = 3  # failed chunks are retried until this many embed attempts

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Idempotent column additions for databases created by older builds."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
        if "attempts" not in columns:
            conn.execute("ALTER TABLE kb_chunks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        source_columns = {row["name"] for row in conn.execute("PRAGMA table_info(kb_sources)")}
        if "messages_done" not in source_columns:
            conn.execute("ALTER TABLE kb_sources ADD COLUMN messages_done INTEGER NOT NULL DEFAULT 0")

    @classmethod
    def default(cls) -> "KbStore":
        return cls(default_db_path())

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_snapshot(
        self,
        *,
        content_hash: str,
        client: str,
        provider: str,
        model: str,
        system_prompt: str,
        developer_prompt: str,
        tools_json: str,
        seen_at: str,
    ) -> tuple[int, bool]:
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
                (content_hash, client, provider, model, system_prompt, developer_prompt, tools_json, seen_at, seen_at),
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
            row = conn.execute("SELECT 1 FROM kb_sources WHERE session_id=?", (session_id,)).fetchone()
            return row is not None

    def record_source(
        self, session_id: str, snapshot_id: int | None, processed_at: str, *, messages_done: bool = False
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kb_sources (session_id, snapshot_id, processed_at, messages_done)"
                " VALUES (?, ?, ?, ?)",
                (session_id, snapshot_id, processed_at, 1 if messages_done else 0),
            )

    def sources_missing_messages(self, limit: int = 50) -> list[str]:
        """Session ids recorded before message extraction existed (or whose
        backfill failed); the lazy loop backfills user messages for these."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM kb_sources WHERE messages_done = 0 LIMIT ?",
                (limit,),
            ).fetchall()
            return [str(row["session_id"]) for row in rows]

    def mark_source_messages_done(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE kb_sources SET messages_done = 1 WHERE session_id = ?",
                (session_id,),
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
                "UPDATE kb_chunks SET index_state='failed', attempts=attempts+1 WHERE id=?",
                (chunk_id,),
            )

    def requeue_failed(self) -> int:
        """Move failed chunks below the attempt cap back to pending so the
        next indexing round retries them; chunks at the cap stay failed and
        remain visible via stats()."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE kb_chunks SET index_state='pending' WHERE index_state='failed' AND attempts < ?",
                (self.MAX_ATTEMPTS,),
            )
            return cur.rowcount

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
            cur = conn.execute("UPDATE kb_chunks SET embedding=NULL, index_state='pending', attempts=0")
            return cur.rowcount

    def upsert_message(
        self,
        *,
        session_id: str,
        record_index: int,
        message_index: int,
        client: str,
        model: str,
        timestamp: str,
        content_hash: str,
        text: str,
        seen_at: str,
    ) -> tuple[int, bool]:
        """Insert a user-message chunk; dedup on (content_hash, client).

        Returns (message_id, created). On dedup hit only last_seen is updated
        and the first-seen session_id is kept.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM kb_messages WHERE content_hash=? AND client=?",
                (content_hash, client),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE kb_messages SET last_seen=? WHERE id=?",
                    (seen_at, row["id"]),
                )
                return int(row["id"]), False
            cur = conn.execute(
                """INSERT INTO kb_messages
                   (session_id, record_index, message_index, client, model,
                    timestamp, content_hash, text, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, record_index, message_index, client, model, timestamp, content_hash, text, seen_at),
            )
            return int(cur.lastrowid), True

    def pending_messages(self, limit: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, session_id, text, last_seen FROM kb_messages WHERE index_state='pending' ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()

    def mark_message_indexed(self, message_id: int, embedding: bytes) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE kb_messages SET embedding=?, index_state='indexed' WHERE id=?",
                (embedding, message_id),
            )

    def mark_message_failed(self, message_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE kb_messages SET index_state='failed', attempts=attempts+1 WHERE id=?",
                (message_id,),
            )

    def requeue_failed_messages(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE kb_messages SET index_state='pending' WHERE index_state='failed' AND attempts < ?",
                (self.MAX_ATTEMPTS,),
            )
            return cur.rowcount

    def indexed_messages(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """SELECT id, session_id, client, model, timestamp, text, embedding
                   FROM kb_messages WHERE index_state='indexed'"""
            ).fetchall()

    def reset_message_embeddings(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("UPDATE kb_messages SET embedding=NULL, index_state='pending', attempts=0")
            return cur.rowcount

    def delete_messages_for_session(self, session_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM kb_messages WHERE session_id=?", (session_id,))
            return cur.rowcount

    def get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kb_meta WHERE key=?", (key,)).fetchone()
            return str(row["value"]) if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES (?, ?)", (key, value))

    def get_snapshot(self, snapshot_id: int) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM kb_snapshots WHERE id=?", (snapshot_id,)).fetchone()

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
                for row in conn.execute("SELECT index_state, COUNT(*) c FROM kb_chunks GROUP BY index_state")
            }
            return {
                "snapshots": int(snapshots),
                "chunks": int(chunks),
                "pending": int(by_state.get("pending", 0)),
                "failed": int(by_state.get("failed", 0)),
                "indexed": int(by_state.get("indexed", 0)),
                "messages": int(conn.execute("SELECT COUNT(*) c FROM kb_messages").fetchone()["c"]),
            }
