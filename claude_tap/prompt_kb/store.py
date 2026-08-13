"""SQLite storage for the prompt knowledge base.

The KB database is derived data: it can always be rebuilt from the trace
store. It lives next to the trace database (CLOUDTAP_DB) so tests that
redirect the trace DB automatically get an isolated KB.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from claude_tap.prompt_kb.chunk import BOILERPLATE_TITLES
from claude_tap.prompt_kb.tokenize import segment
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
  attempts INTEGER NOT NULL DEFAULT 0,
  role TEXT NOT NULL DEFAULT 'user'
);
CREATE INDEX IF NOT EXISTS idx_kb_messages_state ON kb_messages(index_state);
CREATE INDEX IF NOT EXISTS idx_kb_messages_session ON kb_messages(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_messages_dedup ON kb_messages(content_hash, client, role);
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_chunks_tri USING fts5(text, content='', tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_chunks_jieba USING fts5(text, content='', tokenize='unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_messages_tri USING fts5(text, content='', tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_messages_jieba USING fts5(text, content='', tokenize='unicode61');
"""

FTS_ENTITIES = ("chunks", "messages")
_FTS_TABLE_BY_ENTITY = {"chunks": "kb_chunks", "messages": "kb_messages"}


def fts_tables(entity: str) -> tuple[str, str]:
    """(trigram_table, jieba_table) for an entity ("chunks" | "messages")."""
    if entity not in FTS_ENTITIES:
        raise ValueError(f"unknown FTS entity: {entity!r}")
    return (f"kb_fts_{entity}_tri", f"kb_fts_{entity}_jieba")


def _fts_insert(conn: sqlite3.Connection, entity: str, rowid: int, text: str) -> None:
    tri, jieba = fts_tables(entity)
    conn.execute(f"INSERT INTO {tri} (rowid, text) VALUES (?, ?)", (rowid, text))
    conn.execute(f"INSERT INTO {jieba} (rowid, text) VALUES (?, ?)", (rowid, segment(text)))


def _fts_delete(conn: sqlite3.Connection, entity: str, rowid: int, text: str) -> None:
    """Contentless FTS rows are removed via the special 'delete' insert, which
    needs the exact text that was indexed (segmented for the jieba table)."""
    tri, jieba = fts_tables(entity)
    conn.execute(f"INSERT INTO {tri} ({tri}, rowid, text) VALUES ('delete', ?, ?)", (rowid, text))
    conn.execute(f"INSERT INTO {jieba} ({jieba}, rowid, text) VALUES ('delete', ?, ?)", (rowid, segment(text)))


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
        message_columns = {row["name"] for row in conn.execute("PRAGMA table_info(kb_messages)")}
        if "role" not in message_columns:
            conn.execute("ALTER TABLE kb_messages ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
            conn.execute("DROP INDEX IF EXISTS idx_kb_messages_dedup")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_messages_dedup ON kb_messages(content_hash, client, role)"
            )
            # Trigger a full message backfill: assistant replies are net-new,
            # user messages dedup idempotently under the new (hash, client, role) key.
            conn.execute("UPDATE kb_sources SET messages_done = 0")
        purged = conn.execute("SELECT value FROM kb_meta WHERE key='boilerplate_purged'").fetchone()
        if purged is None or purged["value"] != "1":
            placeholders = ",".join("?" for _ in BOILERPLATE_TITLES)
            conn.execute(
                f"DELETE FROM kb_chunks WHERE lower(title) IN ({placeholders})",
                sorted(BOILERPLATE_TITLES),
            )
            conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('boilerplate_purged', '1')")

    @classmethod
    def default(cls) -> "KbStore":
        return cls(default_db_path())

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        # Wait up to 2s on a writer's lock (e.g. the dashboard's lazy indexer)
        # instead of failing reads instantly with "database is locked".
        conn.execute("PRAGMA busy_timeout = 2000")
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
            old = conn.execute("SELECT id, text FROM kb_chunks WHERE snapshot_id=?", (snapshot_id,)).fetchall()
            for row in old:
                _fts_delete(conn, "chunks", row["id"], row["text"])
            conn.execute("DELETE FROM kb_chunks WHERE snapshot_id=?", (snapshot_id,))
            for kind, title, text in chunks:
                cur = conn.execute(
                    "INSERT INTO kb_chunks (snapshot_id, kind, title, text) VALUES (?, ?, ?, ?)",
                    (snapshot_id, kind, title, text),
                )
                _fts_insert(conn, "chunks", int(cur.lastrowid), text)

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
        role: str = "user",
        seen_at: str,
    ) -> tuple[int, bool]:
        """Insert a message chunk; dedup on (content_hash, client, role).

        Returns (message_id, created). On dedup hit only last_seen is updated
        and the first-seen session_id is kept.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM kb_messages WHERE content_hash=? AND client=? AND role=?",
                (content_hash, client, role),
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
                    timestamp, content_hash, text, last_seen, role)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, record_index, message_index, client, model, timestamp, content_hash, text, seen_at, role),
            )
            _fts_insert(conn, "messages", int(cur.lastrowid), text)
            return int(cur.lastrowid), True

    def pending_messages(self, limit: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """SELECT id, session_id, text, last_seen, role
                   FROM kb_messages WHERE index_state='pending' ORDER BY id LIMIT ?""",
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
                """SELECT id, session_id, client, model, timestamp, text, embedding, role
                   FROM kb_messages WHERE index_state='indexed'"""
            ).fetchall()

    def fts_rank(self, entity: str, tokenizer: str, match_query: str, limit: int) -> list[tuple[int, float]]:
        """BM25 ranking over one FTS table: [(rowid, positive score)], best first.

        bm25() is negative (smaller = better); negated here so higher = better.
        A missing table (pre-migration DB) or a bad MATCH yields an empty channel.
        """
        if tokenizer not in ("tri", "jieba"):
            raise ValueError(f"unknown FTS tokenizer: {tokenizer!r}")
        table = fts_tables(entity)[0 if tokenizer == "tri" else 1]
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT rowid, bm25({table}) AS r FROM {table} WHERE {table} MATCH ? ORDER BY r LIMIT ?",
                    (match_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(row["rowid"]), -float(row["r"])) for row in rows]

    def reset_message_embeddings(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("UPDATE kb_messages SET embedding=NULL, index_state='pending', attempts=0")
            return cur.rowcount

    def delete_messages_for_session(self, session_id: str) -> int:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, text FROM kb_messages WHERE session_id=?", (session_id,)).fetchall()
            for row in rows:
                _fts_delete(conn, "messages", row["id"], row["text"])
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
            by_role = {
                row["role"]: row["c"] for row in conn.execute("SELECT role, COUNT(*) c FROM kb_messages GROUP BY role")
            }
            return {
                "snapshots": int(snapshots),
                "chunks": int(chunks),
                "pending": int(by_state.get("pending", 0)),
                "failed": int(by_state.get("failed", 0)),
                "indexed": int(by_state.get("indexed", 0)),
                "messages": int(conn.execute("SELECT COUNT(*) c FROM kb_messages").fetchone()["c"]),
                "messages_user": int(by_role.get("user", 0)),
                "messages_assistant": int(by_role.get("assistant", 0)),
            }
