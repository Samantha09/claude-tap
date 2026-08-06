# Prompt 知识库（RAG 方向 D）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 claude-tap 被动采集的 prompt 快照升级为本地可语义检索的 prompt/工具知识库，dashboard 加"Prompt 知识库"页，`claude-tap kb` CLI 可查。

**Architecture:** 新独立包 `claude_tap/prompt_kb/`（store/chunk/embed/extract/index/search/cli），SQLite + numpy 暴力余弦；dashboard（aiohttp，`live.py`）长驻进程内后台线程懒索引；复用 `prompt_snapshot.snapshot_from_records` 做提取。proxy 录制路径零改动。

**Tech Stack:** Python 3.11+、sqlite3（stdlib）、numpy + sentence-transformers（可选依赖 `[rag]`）、aiohttp（已有）、pytest（已有）。

**Spec:** `docs/superpowers/specs/2026-08-06-prompt-kb-rag-design.md`

## Global Constraints

- Python >= 3.11（沿用 pyproject 现有要求）
- `numpy`、`sentence-transformers` **只允许**出现在 `[project.optional-dependencies] rag` 中，核心安装不得引入
- 未装 `[rag]` 时：trace 录制、dashboard 既有功能零影响；知识库 API 返回 501，页面显示安装提示
- 所有 commit message 用中文（含 type/scope 前缀）；代码与注释用英文
- 测试用仓库现有 pytest 模式与 `trace_db` fixture（`tests/conftest.py:64`）
- 不修改 proxy/trace 录制路径的任何行为
- dashboard 页面 i18n 条目以中文为源语言编写（`DASHBOARD_I18N` 中 zh-CN 在前）
- KB 数据库是派生数据，路径跟随 trace DB：`Path(os.environ["CLOUDTAP_DB"]).with_name("prompt_kb.sqlite3")`；未设环境变量时用 `~/.local/share/claude-tap/prompt_kb.sqlite3`

## 关键既有接口（实现前必读）

- `prompt_snapshot.snapshot_from_records(records: list[dict]) -> PromptSnapshot`；无 prompt 内容时 raise `ValueError`。`PromptSnapshot` 字段：`provider, model, system_prompt, developer_prompt, tools: tuple[PromptTool], captured_at`；`PromptTool` 字段：`name, description, schema`
- `trace_store.TraceStore.list_session_rows(limit=, offset=, query=) -> list[sqlite3.Row]`（row 含 `id, client, started_at, status, record_count`）；`load_records(session_id, limit=, offset=) -> list[dict]`
- `trace_store.reset_trace_store()`：测试用，`trace_db` fixture 已调用
- `cli.py:1055 main_entry()`：按 `sys.argv[1]` 分发子命令（`export`/`update`/`dashboard` 模式）
- `live.py LiveViewerServer`：aiohttp 路由集中在 `start()` 内 `app.router.add_get/post/delete(...)`（约 228-252 行）；handler 形如 `async def _handle_stats(self, request: web.Request) -> web.Response`
- `dashboard.html`：视图切换为 `.view-toggle-btn[data-view]`（现有 `list`/`stats`），对应 `#list-view`/`#stats-view` section 与 `showListView()/showStatsView()`；i18n 用 `DASHBOARD_I18N` dict + `t(key)`
- `tests/conftest.py:64 trace_db`：monkeypatch `CLOUDTAP_DB` 指向临时库

## File Structure

| 文件 | 责任 |
|------|------|
| `claude_tap/prompt_kb/__init__.py` | 公开接口 re-export |
| `claude_tap/prompt_kb/store.py` | SQLite schema + 全部读写（KbStore） |
| `claude_tap/prompt_kb/chunk.py` | PromptSnapshot → Chunk 列表；content_hash |
| `claude_tap/prompt_kb/embed.py` | Embedder 协议、Local/Api 实现、配置加载 |
| `claude_tap/prompt_kb/extract.py` | trace session → snapshot 入库（含 kb_sources 标记） |
| `claude_tap/prompt_kb/index.py` | pending chunks 批量 embedding；rebuild；后台循环 |
| `claude_tap/prompt_kb/search.py` | 查询 → 余弦排序 → 按快照分组 |
| `claude_tap/prompt_kb/cli.py` | `claude-tap kb` 子命令 |
| `tests/prompt_kb/fake_embedder.py` | 确定性假 Embedder（16 维词袋哈希向量） |

---

### Task 1: store.py — SQLite schema 与读写

**Files:**
- Create: `claude_tap/prompt_kb/__init__.py`
- Create: `claude_tap/prompt_kb/store.py`
- Test: `tests/prompt_kb/test_store.py`

**Interfaces:**
- Consumes: 无（只依赖 stdlib sqlite3 与 CLOUDTAP_DB 环境约定）
- Produces（后续所有任务依赖）:
  - `KbStore(db_path: Path)`，`.default() -> KbStore`
  - `.upsert_snapshot(*, content_hash: str, client: str, provider: str, model: str, system_prompt: str, developer_prompt: str, tools_json: str, seen_at: str) -> tuple[int, bool]`（返回 snapshot_id 与是否新建）
  - `.replace_chunks(snapshot_id: int, chunks: list[tuple[str, str, str]]) -> None`（每项为 `(kind, title, text)`，全部置 pending）
  - `.is_source_processed(session_id: str) -> bool`
  - `.record_source(session_id: str, snapshot_id: int | None, processed_at: str) -> None`
  - `.pending_chunks(limit: int) -> list[sqlite3.Row]`（row 含 `id, text`）
  - `.mark_chunk_indexed(chunk_id: int, embedding: bytes) -> None`
  - `.mark_chunk_failed(chunk_id: int) -> None`
  - `.indexed_chunks() -> list[sqlite3.Row]`（row 含 `id, snapshot_id, kind, title, text, embedding`）
  - `.reset_embeddings() -> int`（清空所有 embedding 并置 pending，返回条数）
  - `.get_meta(key: str) -> str | None`，`.set_meta(key: str, value: str) -> None`
  - `.get_snapshot(snapshot_id: int) -> sqlite3.Row | None`
  - `.timeline(client: str, model: str) -> list[sqlite3.Row]`（按 first_seen 升序）
  - `.stats() -> dict`（`{"snapshots": int, "chunks": int, "pending": int, "failed": int, "indexed": int}`）

- [ ] **Step 1: 写失败测试**

```python
# tests/prompt_kb/test_store.py
from claude_tap.prompt_kb.store import KbStore


def _upsert(store, *, content_hash="h1", client="codex", model="gpt-5", seen_at="2026-08-01T00:00:00Z"):
    return store.upsert_snapshot(
        content_hash=content_hash,
        client=client,
        provider="openai",
        model=model,
        system_prompt="sys",
        developer_prompt="",
        tools_json="[]",
        seen_at=seen_at,
    )


def test_upsert_dedups_by_client_model_hash(trace_db):
    store = KbStore.default()
    first_id, created = _upsert(store)
    assert created is True
    again_id, created = _upsert(store, seen_at="2026-08-02T00:00:00Z")
    assert created is False
    assert again_id == first_id
    row = store.get_snapshot(first_id)
    assert row["session_count"] == 2
    assert row["last_seen"] == "2026-08-02T00:00:00Z"
    assert row["first_seen"] == "2026-08-01T00:00:00Z"


def test_timeline_orders_by_first_seen(trace_db):
    store = KbStore.default()
    _upsert(store, content_hash="h2", seen_at="2026-08-02T00:00:00Z")
    _upsert(store, content_hash="h1", seen_at="2026-08-01T00:00:00Z")
    versions = store.timeline("codex", "gpt-5")
    assert [v["content_hash"] for v in versions] == ["h1", "h2"]


def test_chunks_lifecycle(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("tool", "shell", "shell tool desc"), ("prompt_section", "Rules", "rule text")])
    pending = store.pending_chunks(10)
    assert len(pending) == 2
    store.mark_chunk_indexed(pending[0]["id"], b"\x00" * 64)
    store.mark_chunk_failed(pending[1]["id"])
    assert store.stats() == {"snapshots": 1, "chunks": 2, "pending": 0, "failed": 1, "indexed": 1}
    assert len(store.indexed_chunks()) == 1
    assert store.reset_embeddings() == 2
    assert store.stats()["pending"] == 2


def test_sources_mark_processed(trace_db):
    store = KbStore.default()
    assert store.is_source_processed("s1") is False
    store.record_source("s1", None, "2026-08-06T00:00:00Z")
    assert store.is_source_processed("s1") is True


def test_meta_roundtrip(trace_db):
    store = KbStore.default()
    assert store.get_meta("embedder_name") is None
    store.set_meta("embedder_name", "fake")
    assert store.get_meta("embedder_name") == "fake"
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_store.py -q`
Expected: FAIL（`ModuleNotFoundError: claude_tap.prompt_kb`）

- [ ] **Step 3: 实现**

```python
# claude_tap/prompt_kb/__init__.py
"""Local prompt/tool knowledge base over captured trace snapshots."""
```

```python
# claude_tap/prompt_kb/store.py
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
```

注意：`trace_db` fixture 设置 `CLOUDTAP_DB` 后 `KbStore.default()` 自动落到临时目录，测试无需额外 fixture。

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_store.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/__init__.py claude_tap/prompt_kb/store.py tests/prompt_kb/test_store.py
git commit -m "feat(prompt_kb): 新增知识库 SQLite 存储层"
```

---

### Task 2: chunk.py — 切块与 content_hash

**Files:**
- Create: `claude_tap/prompt_kb/chunk.py`
- Test: `tests/prompt_kb/test_chunk.py`

**Interfaces:**
- Consumes: `prompt_snapshot.PromptSnapshot` / `PromptTool`
- Produces:
  - `Chunk` dataclass：`kind: str, title: str, text: str`（`kind` 为 `"prompt_section"` 或 `"tool"`）
  - `chunk_snapshot(snapshot: PromptSnapshot) -> list[Chunk]`
  - `content_hash(client: str, model: str, snapshot: PromptSnapshot) -> str`
  - 供 Task 4 把 `Chunk` 转成 store 元组：`(c.kind, c.title, c.text)`

规则（与 spec 一致）：system/developer prompt 按 Markdown 标题切；无标题按段落合并；单块上限 `MAX_SECTION_CHARS = 2000`（约 500 token），小于 `MIN_SECTION_CHARS = 200` 的块并入前一块；每个 tool 一条，text = `name + "\n" + description + "\n参数: " + 参数名列表`（参数名取 schema `properties` 的 key）；content_hash 对 prompt 做行尾空白归一、tools 按 name 排序后规范序列化，sha256。

- [ ] **Step 1: 写失败测试**

```python
# tests/prompt_kb/test_chunk.py
from claude_tap.prompt_kb.chunk import chunk_snapshot, content_hash
from claude_tap.prompt_snapshot import PromptSnapshot, PromptTool


def _snapshot(system="", developer="", tools=()):
    return PromptSnapshot(
        provider="anthropic", model="claude", system_prompt=system,
        developer_prompt=developer, tools=tools,
    )


def test_splits_by_markdown_headings():
    snap = _snapshot(system="# Rules\nbe nice\n# Tools\nuse them wisely")
    chunks = chunk_snapshot(snap)
    assert [c.title for c in chunks] == ["Rules", "Tools"]
    assert chunks[0].text == "# Rules\nbe nice"
    assert all(c.kind == "prompt_section" for c in chunks)


def test_merges_tiny_sections():
    snap = _snapshot(system="# A\nhi\n# B\n" + "long text " * 50)
    chunks = chunk_snapshot(snap)
    assert len(chunks) == 1
    assert chunks[0].title == "B"


def test_splits_long_section_without_headings():
    para = "word " * 300
    snap = _snapshot(system=f"{para}\n\n{para}")
    chunks = chunk_snapshot(snap)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 2000 for c in chunks)


def test_tool_chunk_format():
    tool = PromptTool(
        name="shell",
        description="run commands",
        schema={"input_schema": {"properties": {"cmd": {}, "timeout": {}}}},
    )
    chunks = chunk_snapshot(_snapshot(tools=(tool,)))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.kind == "tool" and c.title == "shell"
    assert c.text.startswith("shell\nrun commands")
    assert "cmd" in c.text and "timeout" in c.text


def test_content_hash_stable_against_whitespace_and_tool_order():
    t1 = PromptTool(name="b", description="2", schema={})
    t2 = PromptTool(name="a", description="1", schema={})
    snap_a = _snapshot(system="line one  \nline two", tools=(t1, t2))
    snap_b = _snapshot(system="line one\nline two", tools=(t2, t1))
    assert content_hash("codex", "gpt-5", snap_a) == content_hash("codex", "gpt-5", snap_b)


def test_content_hash_changes_with_content():
    assert content_hash("codex", "gpt-5", _snapshot(system="a")) != content_hash(
        "codex", "gpt-5", _snapshot(system="b")
    )
```

tool schema 的 properties 兼容 anthropic（`input_schema.properties`）与 openai（`parameters.properties`）两种包装。

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_chunk.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# claude_tap/prompt_kb/chunk.py
"""Split prompt snapshots into embeddable chunks and compute content hashes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from claude_tap.prompt_snapshot import PromptSnapshot, PromptTool

MAX_SECTION_CHARS = 2000
MIN_SECTION_CHARS = 200

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    kind: str  # "prompt_section" | "tool"
    title: str
    text: str


def chunk_snapshot(snapshot: PromptSnapshot) -> list[Chunk]:
    chunks: list[Chunk] = []
    for label, prompt in (("system", snapshot.system_prompt),
                          ("developer", snapshot.developer_prompt)):
        if prompt and prompt.strip():
            chunks.extend(_split_prompt(prompt))
    chunks.extend(_tool_chunk(tool) for tool in snapshot.tools)
    return chunks


def _split_prompt(text: str) -> list[Chunk]:
    sections = _heading_sections(text.strip())
    merged = _merge_small(sections)
    chunks: list[Chunk] = []
    for title, body in merged:
        for piece in _split_long(title, body):
            chunks.append(Chunk(kind="prompt_section", title=piece[0], text=piece[1]))
    return chunks


def _heading_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings; text before the first heading is one section."""
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current_lines or current_title:
                sections.append((current_title, current_lines))
            current_title = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines or current_title:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(lines).strip()) for title, lines in sections
            if "\n".join(lines).strip()]


def _merge_small(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for title, body in sections:
        if merged and len(body) < MIN_SECTION_CHARS:
            prev_title, prev_body = merged[-1]
            merged[-1] = (title or prev_title, prev_body + "\n\n" + body)
        else:
            merged.append((title, body))
    return merged


def _split_long(title: str, body: str) -> list[tuple[str, str]]:
    if len(body) <= MAX_SECTION_CHARS:
        return [(title, body)]
    pieces: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0
    for para in re.split(r"\n\s*\n", body):
        if current and current_len + len(para) + 2 > MAX_SECTION_CHARS:
            pieces.append((title, "\n\n".join(current)))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        pieces.append((title, "\n\n".join(current)))
    # Hard-split any paragraph that alone exceeds the limit.
    result: list[tuple[str, str]] = []
    for piece_title, piece in pieces:
        while len(piece) > MAX_SECTION_CHARS:
            result.append((piece_title, piece[:MAX_SECTION_CHARS]))
            piece = piece[MAX_SECTION_CHARS:]
        result.append((piece_title, piece))
    return result


def _tool_chunk(tool: PromptTool) -> Chunk:
    params = _tool_param_names(tool.schema)
    text = tool.name
    if tool.description:
        text += "\n" + tool.description
    if params:
        text += "\n参数: " + ", ".join(params)
    return Chunk(kind="tool", title=tool.name, text=text)


def _tool_param_names(schema: dict[str, Any]) -> list[str]:
    for key in ("input_schema", "parameters"):
        wrapper = schema.get(key)
        if isinstance(wrapper, dict):
            props = wrapper.get("properties")
            if isinstance(props, dict):
                return sorted(str(name) for name in props)
    props = schema.get("properties")
    if isinstance(props, dict):
        return sorted(str(name) for name in props)
    return []


def content_hash(client: str, model: str, snapshot: PromptSnapshot) -> str:
    payload = {
        "client": client,
        "model": model,
        "system": _normalize_text(snapshot.system_prompt),
        "developer": _normalize_text(snapshot.developer_prompt),
        "tools": [
            {"name": tool.name, "description": tool.description,
             "params": _tool_param_names(tool.schema)}
            for tool in sorted(snapshot.tools, key=lambda t: t.name)
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_chunk.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/chunk.py tests/prompt_kb/test_chunk.py
git commit -m "feat(prompt_kb): 新增快照切块与内容 hash"
```

---

### Task 3: embed.py — Embedder 抽象、本地/API 实现、配置

**Files:**
- Create: `claude_tap/prompt_kb/embed.py`
- Create: `tests/prompt_kb/fake_embedder.py`
- Test: `tests/prompt_kb/test_embed.py`

**Interfaces:**
- Consumes: 无（只依赖 stdlib；sentence-transformers 延迟 import）
- Produces:
  - `class EmbedderUnavailable(Exception)`
  - `class Embedder(Protocol)`：属性 `name: str`、`dimension: int`；方法 `embed(texts: list[str]) -> list[list[float]]`
  - `KbConfig` dataclass：`embedder: str`（`"local"|"api"`）、`local_model: str`、`api_base: str`、`api_model: str`、`api_key_env: str`
  - `load_config(path: Path | None = None) -> KbConfig`（读 toml `[prompt_kb]`，环境变量 `CLAUDE_TAP_KB_*` 覆盖）
  - `create_embedder(config: KbConfig) -> Embedder`（缺依赖/缺配置时 raise `EmbedderUnavailable`）
  - `vectors_to_blob(vectors: list[list[float]]) -> list[bytes]`（float32 小端，供 store 写入；不依赖 numpy——用 `struct`/`array`）
  - 测试用 `tests/prompt_kb/fake_embedder.py`：`FakeEmbedder(dimension=16)`，`embed` 把词袋哈希进固定维度并 L2 归一（共同词越多余弦越高），`name="fake"`、`dimension=16`

- [ ] **Step 1: 写失败测试**

```python
# tests/prompt_kb/test_embed.py
import math

import pytest

from claude_tap.prompt_kb import embed as embed_mod
from claude_tap.prompt_kb.embed import (
    EmbedderUnavailable,
    KbConfig,
    create_embedder,
    load_config,
    vectors_to_blob,
)
from tests.prompt_kb.fake_embedder import FakeEmbedder


def test_load_config_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_TAP_KB_EMBEDDER", raising=False)
    config = load_config(tmp_path / "missing.toml")
    assert config.embedder == "local"
    assert config.local_model == "intfloat/multilingual-e5-small"


def test_load_config_toml_and_env_override(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[prompt_kb]\nembedder = "api"\napi_base = "https://x.example/v1"\napi_model = "m"\n')
    monkeypatch.setenv("CLAUDE_TAP_KB_EMBEDDER", "local")
    config = load_config(path)
    assert config.embedder == "local"          # env wins
    assert config.api_base == "https://x.example/v1"


def test_create_embedder_local_without_dependency(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
    with pytest.raises(EmbedderUnavailable):
        create_embedder(KbConfig(embedder="local"))


def test_create_embedder_api_requires_key(monkeypatch):
    monkeypatch.delenv("KB_TEST_KEY", raising=False)
    with pytest.raises(EmbedderUnavailable):
        create_embedder(KbConfig(
            embedder="api", api_base="https://x.example/v1",
            api_model="m", api_key_env="KB_TEST_KEY",
        ))


def test_vectors_to_blob_roundtrip():
    blobs = vectors_to_blob([[1.0, 0.5], [0.0, 2.0]])
    import array
    values = array.array("f")
    values.frombytes(blobs[0])
    assert list(values) == [1.0, 0.5]
    assert len(blobs[0]) == 8  # 2 * float32


def test_fake_embedder_semantic():
    emb = FakeEmbedder()
    a = emb.embed(["sandbox shell command"])[0]
    b = emb.embed(["shell command runner"])[0]
    c = emb.embed(["totally unrelated topic"])[0]
    def cos(x, y):
        return sum(p * q for p, q in zip(x, y)) / (
            math.sqrt(sum(p * p for p in x)) * math.sqrt(sum(q * q for q in y)))
    assert cos(a, b) > cos(a, c)
    assert len(a) == 16
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_embed.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# tests/prompt_kb/__init__.py
```

```python
# tests/prompt_kb/fake_embedder.py
"""Deterministic bag-of-words hash embedder for tests (no model download)."""

from __future__ import annotations

import hashlib
import math
import re


class FakeEmbedder:
    name = "fake"
    dimension = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vec[digest[0] % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
```

```python
# claude_tap/prompt_kb/embed.py
"""Embedder abstraction: local sentence-transformers by default, API optional.

Configuration comes from `[prompt_kb]` in the config TOML (default path
`~/.config/claude-tap/config.toml`), overridable per-key with
`CLAUDE_TAP_KB_*` environment variables.
"""

from __future__ import annotations

import array
import json
import os
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "claude-tap" / "config.toml"


class EmbedderUnavailable(Exception):
    """Raised when no usable embedder is configured or installed."""


class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class KbConfig:
    embedder: str = "local"  # "local" | "api"
    local_model: str = DEFAULT_LOCAL_MODEL
    api_base: str = ""
    api_model: str = ""
    api_key_env: str = "OPENAI_API_KEY"


def load_config(path: Path | None = None) -> KbConfig:
    values: dict[str, str] = {}
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.is_file():
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        section = data.get("prompt_kb") or {}
        values.update({k: str(v) for k, v in section.items()})
    for key in ("embedder", "local_model", "api_base", "api_model", "api_key_env"):
        env = os.environ.get(f"CLAUDE_TAP_KB_{key.upper()}")
        if env:
            values[key] = env
    known = {f for f in KbConfig.__dataclass_fields__}
    return KbConfig(**{k: v for k, v in values.items() if k in known})


class LocalEmbedder:
    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderUnavailable(
                "sentence-transformers is not installed; "
                "install the optional dependency: pip install 'claude-tap[rag]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = f"local:{model_name}"
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vec)) for vec in vectors]


class ApiEmbedder:
    """OpenAI-compatible /embeddings client over stdlib urllib."""

    def __init__(self, *, api_base: str, api_model: str, api_key: str):
        self._endpoint = api_base.rstrip("/") + "/embeddings"
        self._model = api_model
        self._api_key = api_key
        self.name = f"api:{api_model}"
        self.dimension = 0  # unknown until first response

    def embed(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps({"model": self._model, "input": texts}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        vectors = [list(map(float, item["embedding"])) for item in payload["data"]]
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors


def create_embedder(config: KbConfig) -> Embedder:
    if config.embedder == "api":
        if not config.api_base or not config.api_model:
            raise EmbedderUnavailable("api embedder requires api_base and api_model")
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise EmbedderUnavailable(
                f"api embedder requires the {config.api_key_env} environment variable"
            )
        return ApiEmbedder(api_base=config.api_base, api_model=config.api_model, api_key=api_key)
    return LocalEmbedder(config.local_model)


def vectors_to_blob(vectors: list[list[float]]) -> list[bytes]:
    blobs = []
    for vec in vectors:
        arr = array.array("f", vec)
        blobs.append(arr.tobytes())
    return blobs
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_embed.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/embed.py tests/prompt_kb/__init__.py tests/prompt_kb/fake_embedder.py tests/prompt_kb/test_embed.py
git commit -m "feat(prompt_kb): 新增 Embedder 抽象与本地/API 实现"
```

---

### Task 4: extract.py — trace session → snapshot 入库

**Files:**
- Create: `claude_tap/prompt_kb/extract.py`
- Test: `tests/prompt_kb/test_extract.py`

**Interfaces:**
- Consumes: `KbStore`（Task 1）、`chunk_snapshot`/`content_hash`（Task 2）、`prompt_snapshot.snapshot_from_records`、`trace_store.TraceStore`
- Produces:
  - `extract_session(store: KbStore, *, session_id: str, client: str, records: list[dict], processed_at: str) -> int | None`（返回 snapshot_id；无 prompt 内容或提取失败记 `kb_sources(snapshot_id=None)` 并返回 None）
  - `extract_unprocessed(store: KbStore, trace: TraceStore, *, limit: int = 50) -> dict`（`{"processed": int, "snapshots": int, "skipped": int}`，跳过 `record_count == 0` 与已处理会话）

- [ ] **Step 1: 写失败测试**

需要一个带 prompt 的 anthropic 记录 fixture（参照 `tests/test_dashboard.py:68 _anthropic_record` 的结构：`request.path="/v1/messages"`、body 含 `system` 与 `tools`）：

```python
# tests/prompt_kb/test_extract.py
from claude_tap.prompt_kb.extract import extract_session, extract_unprocessed
from claude_tap.prompt_kb.store import KbStore
from claude_tap.trace_store import get_trace_store


def _anthropic_record(turn: int = 1) -> dict:
    return {
        "timestamp": f"2026-08-01T00:00:0{turn}Z",
        "turn": turn,
        "request": {
            "method": "POST",
            "path": "/v1/messages",
            "body": {
                "model": "claude-test",
                "system": "# Rules\nbe helpful",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "shell", "description": "run commands",
                           "input_schema": {"properties": {"cmd": {}}}}],
            },
        },
        "response": {"status": 200, "body": "", "sse_events": []},
    }


def test_extract_session_stores_snapshot_and_chunks(trace_db):
    store = KbStore.default()
    snap_id = extract_session(
        store, session_id="s1", client="claude-code",
        records=[_anthropic_record()], processed_at="2026-08-01T00:00:10Z",
    )
    assert snap_id is not None
    row = store.get_snapshot(snap_id)
    assert row["client"] == "claude-code"
    assert row["model"] == "claude-test"
    assert "Rules" in row["system_prompt"]
    assert store.stats()["pending"] == 2  # 1 prompt section + 1 tool
    assert store.is_source_processed("s1") is True


def test_extract_session_marks_empty_when_no_prompt(trace_db):
    store = KbStore.default()
    snap_id = extract_session(
        store, session_id="s2", client="codex",
        records=[{"request": {"path": "/health", "body": {}}}],
        processed_at="2026-08-01T00:00:10Z",
    )
    assert snap_id is None
    assert store.is_source_processed("s2") is True


def test_extract_unprocessed_walks_trace_store(trace_db):
    trace = get_trace_store()
    session_id = trace.create_session(client="claude-code", proxy_mode="reverse")
    trace.append_record(session_id, _anthropic_record())
    store = KbStore.default()
    result = extract_unprocessed(store, trace)
    assert result == {"processed": 1, "snapshots": 1, "skipped": 0}
    assert extract_unprocessed(store, trace) == {"processed": 0, "snapshots": 0, "skipped": 0}
```

（`TraceStore.create_session(client=..., proxy_mode=...)` 返回 session_id，`get_trace_store()` 单例跟随 `trace_db` fixture 的 `CLOUDTAP_DB`。）

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_extract.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# claude_tap/prompt_kb/extract.py
"""Extract prompt snapshots from trace sessions into the KB store."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from claude_tap.prompt_kb.chunk import chunk_snapshot, content_hash
from claude_tap.prompt_kb.store import KbStore
from claude_tap.prompt_snapshot import snapshot_from_records
from claude_tap.trace_store import TraceStore

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_session(store: KbStore, *, session_id: str, client: str,
                    records: list[dict[str, Any]], processed_at: str) -> int | None:
    try:
        snapshot = snapshot_from_records(records)
    except (ValueError, KeyError, TypeError) as exc:
        logger.info("no prompt snapshot in session %s: %s", session_id, exc)
        store.record_source(session_id, None, processed_at)
        return None
    tools_json = json.dumps(
        [{"name": t.name, "description": t.description, "schema": t.schema}
         for t in snapshot.tools],
        ensure_ascii=False,
    )
    snapshot_id, created = store.upsert_snapshot(
        content_hash=content_hash(client, snapshot.model, snapshot),
        client=client,
        provider=snapshot.provider,
        model=snapshot.model,
        system_prompt=snapshot.system_prompt,
        developer_prompt=snapshot.developer_prompt,
        tools_json=tools_json,
        seen_at=snapshot.captured_at or processed_at,
    )
    if created:
        chunks = chunk_snapshot(snapshot)
        store.replace_chunks(
            snapshot_id, [(c.kind, c.title, c.text) for c in chunks]
        )
    store.record_source(session_id, snapshot_id, processed_at)
    return snapshot_id


def extract_unprocessed(store: KbStore, trace: TraceStore, *, limit: int = 50) -> dict:
    processed = snapshots = skipped = 0
    rows = trace.list_session_rows(limit=limit * 4)  # over-fetch: some are filtered below
    for row in rows:
        if processed >= limit:
            break
        session_id = str(row["id"])
        if store.is_source_processed(session_id):
            continue
        if not row["record_count"]:
            store.record_source(session_id, None, _now())
            skipped += 1
            continue
        records = trace.load_records(session_id)
        snap_id = extract_session(
            store, session_id=session_id,
            client=str(row["client"] or "unknown"),
            records=records, processed_at=_now(),
        )
        processed += 1
        if snap_id is not None:
            snapshots += 1
    return {"processed": processed, "snapshots": snapshots, "skipped": skipped}
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_extract.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/extract.py tests/prompt_kb/test_extract.py
git commit -m "feat(prompt_kb): 新增 trace 会话快照提取管线"
```

---

### Task 5: index.py — 懒索引、rebuild、后台循环

**Files:**
- Create: `claude_tap/prompt_kb/index.py`
- Test: `tests/prompt_kb/test_index.py`

**Interfaces:**
- Consumes: `KbStore`、`Embedder`/`vectors_to_blob`、`extract_unprocessed`
- Produces:
  - `index_pending(store: KbStore, embedder: Embedder, *, batch_size: int = 32) -> dict`（`{"indexed": int, "failed": int, "remaining": int}`；单批 embed 抛错时整批标 failed 并继续下一批）
  - `rebuild_index(store: KbStore, embedder: Embedder) -> dict`（`reset_embeddings()` 后循环 `index_pending` 至无 pending，返回累计计数）
  - `run_index_loop(*, interval_seconds: float = 30.0, stop_event: threading.Event) -> None`（后台线程入口：自建 `KbStore.default()` + `TraceStore()`，每轮 `extract_unprocessed` → `index_pending`；embedder 首次创建失败则每 10 轮重试一次；所有异常 log 后不退出线程）
  - `ensure_embedder_meta(store: KbStore, embedder: Embedder) -> None`（写入 `kb_meta` 的 `embedder_name`/`embedding_dim`）

- [ ] **Step 1: 写失败测试**

```python
# tests/prompt_kb/test_index.py
import array

from claude_tap.prompt_kb.index import index_pending, rebuild_index, ensure_embedder_meta
from claude_tap.prompt_kb.store import KbStore
from tests.prompt_kb.fake_embedder import FakeEmbedder


def _seed(store: KbStore) -> None:
    snap_id, _ = store.upsert_snapshot(
        content_hash="h", client="codex", provider="openai", model="gpt-5",
        system_prompt="s", developer_prompt="", tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(snap_id, [("tool", "a", "alpha text"), ("tool", "b", "beta text")])


def test_index_pending_embeds_and_marks(trace_db):
    store = KbStore.default()
    _seed(store)
    result = index_pending(store, FakeEmbedder())
    assert result == {"indexed": 2, "failed": 0, "remaining": 0}
    chunks = store.indexed_chunks()
    assert len(chunks) == 2
    values = array.array("f")
    values.frombytes(chunks[0]["embedding"])
    assert len(values) == 16


class _FlakyEmbedder(FakeEmbedder):
    def embed(self, texts):
        raise RuntimeError("boom")


def test_index_pending_marks_failed_and_continues(trace_db):
    store = KbStore.default()
    _seed(store)
    result = index_pending(store, _FlakyEmbedder(), batch_size=1)
    assert result == {"indexed": 0, "failed": 2, "remaining": 0}
    assert store.stats()["failed"] == 2


def test_rebuild_resets_then_reindexes(trace_db):
    store = KbStore.default()
    _seed(store)
    index_pending(store, FakeEmbedder())
    assert rebuild_index(store, FakeEmbedder())["indexed"] == 2
    assert store.stats()["indexed"] == 2


def test_ensure_embedder_meta(trace_db):
    store = KbStore.default()
    ensure_embedder_meta(store, FakeEmbedder())
    assert store.get_meta("embedder_name") == "fake"
    assert store.get_meta("embedding_dim") == "16"
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_index.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# claude_tap/prompt_kb/index.py
"""Lazy indexing: embed pending chunks in a long-lived dashboard thread."""

from __future__ import annotations

import logging
import threading

from claude_tap.prompt_kb.embed import Embedder, EmbedderUnavailable, create_embedder, load_config, vectors_to_blob
from claude_tap.prompt_kb.extract import extract_unprocessed
from claude_tap.prompt_kb.store import KbStore
from claude_tap.trace_store import get_trace_store

logger = logging.getLogger(__name__)


def index_pending(store: KbStore, embedder: Embedder, *, batch_size: int = 32) -> dict:
    indexed = failed = 0
    while True:
        batch = store.pending_chunks(batch_size)
        if not batch:
            break
        try:
            vectors = embedder.embed([row["text"] for row in batch])
        except Exception as exc:  # noqa: BLE001 - one bad batch must not stop indexing
            logger.warning("embedding batch failed: %s", exc)
            for row in batch:
                store.mark_chunk_failed(row["id"])
            failed += len(batch)
            continue
        for row, blob in zip(batch, vectors_to_blob(vectors)):
            store.mark_chunk_indexed(row["id"], blob)
        indexed += len(batch)
    return {"indexed": indexed, "failed": failed, "remaining": store.stats()["pending"]}


def rebuild_index(store: KbStore, embedder: Embedder) -> dict:
    store.reset_embeddings()
    return index_pending(store, embedder)


def ensure_embedder_meta(store: KbStore, embedder: Embedder) -> None:
    store.set_meta("embedder_name", embedder.name)
    store.set_meta("embedding_dim", str(embedder.dimension))


def run_index_loop(*, interval_seconds: float = 30.0,
                   stop_event: threading.Event | None = None) -> None:
    """Background entry point for the dashboard process.

    Owns its own KbStore/TraceStore connections. Embedder creation is retried
    every 10 rounds when unavailable (e.g. model not downloaded yet).
    """
    stop = stop_event or threading.Event()
    store = KbStore.default()
    embedder: Embedder | None = None
    rounds_since_embedder_failure = 0
    while not stop.is_set():
        try:
            extract_unprocessed(store, get_trace_store())
            if embedder is None:
                if rounds_since_embedder_failure % 10 == 0:
                    try:
                        embedder = create_embedder(load_config())
                        ensure_embedder_meta(store, embedder)
                    except EmbedderUnavailable as exc:
                        logger.info("kb embedder unavailable: %s", exc)
                rounds_since_embedder_failure += 1
            if embedder is not None:
                index_pending(store, embedder)
        except Exception:  # noqa: BLE001 - the loop must never die silently
            logger.exception("kb index loop iteration failed")
        stop.wait(interval_seconds)
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_index.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/index.py tests/prompt_kb/test_index.py
git commit -m "feat(prompt_kb): 新增懒索引与全量重建"
```

---

### Task 6: search.py — 余弦检索与快照分组

**Files:**
- Create: `claude_tap/prompt_kb/search.py`
- Test: `tests/prompt_kb/test_search.py`

**Interfaces:**
- Consumes: `KbStore`、`Embedder`
- Produces:
  - `class ReindexRequired(Exception)`（kb_meta 的 embedder_name/embedding_dim 与当前 embedder 不符时抛出）
  - `SearchHit` dataclass：`kind: str, title: str, text: str, score: float`
  - `SnapshotResult` dataclass：`snapshot_id: int, client: str, model: str, first_seen: str, last_seen: str, session_count: int, hits: list[SearchHit]`
  - `search(store: KbStore, embedder: Embedder, query: str, *, client: str | None = None, kind: str | None = None, limit: int = 10) -> list[SnapshotResult]`（按组内最高分降序，组内 hits 按分数降序、最多 3 条）
  - 余弦用 numpy；numpy 缺失时 raise `EmbedderUnavailable("numpy is not installed; pip install 'claude-tap[rag]'")`

- [ ] **Step 1: 写失败测试**

```python
# tests/prompt_kb/test_search.py
import pytest

numpy = pytest.importorskip("numpy")  # search depends on the [rag] extra

from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending
from claude_tap.prompt_kb.search import ReindexRequired, search
from claude_tap.prompt_kb.store import KbStore
from tests.prompt_kb.fake_embedder import FakeEmbedder


def _seed(store: KbStore) -> None:
    a, _ = store.upsert_snapshot(
        content_hash="ha", client="codex", provider="openai", model="gpt-5",
        system_prompt="s", developer_prompt="", tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(a, [("tool", "shell", "sandbox shell command runner")])
    b, _ = store.upsert_snapshot(
        content_hash="hb", client="claude-code", provider="anthropic", model="claude",
        system_prompt="s", developer_prompt="", tools_json="[]",
        seen_at="2026-08-02T00:00:00Z",
    )
    store.replace_chunks(b, [("prompt_section", "Style", "write elegant prose")])


def _indexed_store() -> KbStore:
    store = KbStore.default()
    _seed(store)
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    return store


def test_search_returns_grouped_ranked_results(trace_db):
    store = _indexed_store()
    results = search(store, FakeEmbedder(), "shell sandbox")
    assert results[0].client == "codex"
    assert results[0].hits[0].title == "shell"
    assert results[0].hits[0].score > 0.5


def test_search_filters_by_client_and_kind(trace_db):
    store = _indexed_store()
    assert search(store, FakeEmbedder(), "shell", client="claude-code") == []
    results = search(store, FakeEmbedder(), "prose", kind="prompt_section")
    assert all(h.kind == "prompt_section" for r in results for h in r.hits)


def test_search_empty_index_returns_empty(trace_db):
    assert search(KbStore.default(), FakeEmbedder(), "anything") == []


def test_search_detects_embedder_mismatch(trace_db):
    store = _indexed_store()

    class OtherEmbedder(FakeEmbedder):
        name = "other"

    with pytest.raises(ReindexRequired):
        search(store, OtherEmbedder(), "shell")
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_search.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# claude_tap/prompt_kb/search.py
"""Cosine-similarity search over indexed chunks, grouped by snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field

from claude_tap.prompt_kb.embed import Embedder, EmbedderUnavailable
from claude_tap.prompt_kb.store import KbStore


class ReindexRequired(Exception):
    """Stored embeddings were produced by a different embedder."""


@dataclass(frozen=True)
class SearchHit:
    kind: str
    title: str
    text: str
    score: float


@dataclass
class SnapshotResult:
    snapshot_id: int
    client: str
    model: str
    first_seen: str
    last_seen: str
    session_count: int
    hits: list[SearchHit] = field(default_factory=list)


def _check_embedder_meta(store: KbStore, embedder: Embedder) -> None:
    name = store.get_meta("embedder_name")
    dim = store.get_meta("embedding_dim")
    if name is None:
        return  # nothing indexed yet
    if name != embedder.name or (dim and embedder.dimension and dim != str(embedder.dimension)):
        raise ReindexRequired(
            f"indexed with {name} (dim {dim}), current embedder is "
            f"{embedder.name} (dim {embedder.dimension}); run `claude-tap kb reindex`"
        )


def search(store: KbStore, embedder: Embedder, query: str, *,
           client: str | None = None, kind: str | None = None,
           limit: int = 10) -> list[SnapshotResult]:
    _check_embedder_meta(store, embedder)
    rows = store.indexed_chunks()
    if client:
        rows = [row for row in rows if row["client"] == client]
    if kind:
        rows = [row for row in rows if row["kind"] == kind]
    if not rows:
        return []
    try:
        import numpy as np
    except ImportError as exc:
        raise EmbedderUnavailable(
            "numpy is not installed; install the optional dependency: "
            "pip install 'claude-tap[rag]'"
        ) from exc
    matrix = np.array(
        [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
    )
    query_vec = np.array(embedder.embed([query])[0], dtype=np.float32)
    q_norm = np.linalg.norm(query_vec) or 1.0
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms[m_norms == 0] = 1.0
    scores = (matrix @ query_vec) / (m_norms * q_norm)

    groups: dict[int, SnapshotResult] = {}
    for row, score in zip(rows, scores):
        group = groups.setdefault(
            row["snapshot_id"],
            SnapshotResult(
                snapshot_id=row["snapshot_id"], client=row["client"],
                model=row["model"], first_seen=row["first_seen"],
                last_seen=row["last_seen"], session_count=row["session_count"],
            ),
        )
        group.hits.append(SearchHit(
            kind=row["kind"], title=row["title"] or "",
            text=row["text"], score=float(score),
        ))
    ordered = sorted(
        groups.values(),
        key=lambda g: max((h.score for h in g.hits), default=0.0),
        reverse=True,
    )
    for group in ordered:
        group.hits = sorted(group.hits, key=lambda h: h.score, reverse=True)[:3]
    return ordered[:limit]
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_search.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/search.py tests/prompt_kb/test_search.py
git commit -m "feat(prompt_kb): 新增余弦相似度检索与快照分组"
```

---

### Task 7: `claude-tap kb` CLI

**Files:**
- Create: `claude_tap/prompt_kb/cli.py`
- Modify: `claude_tap/cli.py:1055-1099`（`main_entry` 加 `kb` 分发）
- Test: `tests/prompt_kb/test_cli.py`

**Interfaces:**
- Consumes: `KbStore`、`load_config`/`create_embedder`、`index_pending`/`rebuild_index`、`search`
- Produces:
  - `kb_main(argv: list[str]) -> int`：子命令 `search <query> [--client X] [--kind tool|prompt_section] [--limit N]`、`reindex`、`status`；embedder 不可用时打印提示返回 2；`ReindexRequired` 打印提示返回 3

- [ ] **Step 1: 写失败测试**

```python
# tests/prompt_kb/test_cli.py
import pytest

from claude_tap.prompt_kb.cli import kb_main
from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending
from claude_tap.prompt_kb.store import KbStore
from tests.prompt_kb.fake_embedder import FakeEmbedder


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
    monkeypatch.setattr("claude_tap.prompt_kb.cli.create_embedder", lambda config: embedder)
    return store


def test_kb_search_prints_grouped_results(seeded_kb, capsys):
    assert kb_main(["search", "shell sandbox"]) == 0
    out = capsys.readouterr().out
    assert "codex" in out and "gpt-5" in out and "shell" in out


def test_kb_status_prints_counts(seeded_kb, capsys):
    assert kb_main(["status"]) == 0
    out = capsys.readouterr().out
    assert "indexed=1" in out


def test_kb_reindex(seeded_kb, capsys):
    assert kb_main(["reindex"]) == 0
    assert "indexed=1" in capsys.readouterr().out


def test_kb_search_embedder_unavailable(trace_db, monkeypatch, capsys):
    from claude_tap.prompt_kb.embed import EmbedderUnavailable

    def _raise(config):
        raise EmbedderUnavailable("no model")

    monkeypatch.setattr("claude_tap.prompt_kb.cli.create_embedder", _raise)
    assert kb_main(["search", "x"]) == 2
    assert "no model" in capsys.readouterr().err
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_cli.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# claude_tap/prompt_kb/cli.py
"""`claude-tap kb` subcommand: search / reindex / status for the prompt KB."""

from __future__ import annotations

import argparse
import sys

from claude_tap.prompt_kb.embed import EmbedderUnavailable, create_embedder, load_config
from claude_tap.prompt_kb.index import ensure_embedder_meta, rebuild_index
from claude_tap.prompt_kb.search import ReindexRequired, search
from claude_tap.prompt_kb.store import KbStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claude-tap kb")
    sub = parser.add_subparsers(dest="command", required=True)
    search_parser = sub.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--client")
    search_parser.add_argument("--kind", choices=["tool", "prompt_section"])
    search_parser.add_argument("--limit", type=int, default=10)
    sub.add_parser("reindex")
    sub.add_parser("status")
    return parser


def _embedder_or_exit():
    try:
        return create_embedder(load_config())
    except EmbedderUnavailable as exc:
        print(f"kb embedder unavailable: {exc}", file=sys.stderr)
        raise SystemExit(2)


def kb_main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    store = KbStore.default()
    if args.command == "status":
        stats = store.stats()
        print(" ".join(f"{key}={value}" for key, value in stats.items()))
        print(f"embedder={store.get_meta('embedder_name') or 'none'}")
        return 0
    embedder = _embedder_or_exit()
    if args.command == "reindex":
        ensure_embedder_meta(store, embedder)
        result = rebuild_index(store, embedder)
        print(f"indexed={result['indexed']} failed={result['failed']}")
        return 0
    try:
        results = search(store, embedder, args.query,
                         client=args.client, kind=args.kind, limit=args.limit)
    except ReindexRequired as exc:
        print(str(exc), file=sys.stderr)
        return 3
    for rank, group in enumerate(results, 1):
        print(f"[{rank}] {group.client} / {group.model} "
              f"(first seen {group.first_seen}, sessions {group.session_count})")
        for hit in group.hits:
            print(f"    {hit.kind} {hit.title} score={hit.score:.3f}")
            print(f"    {hit.text[:200]}")
    return 0
```

`claude_tap/cli.py` 的 `main_entry()` 中，`if len(sys.argv) > 1 and sys.argv[1] == "dashboard":` 分支之前插入：

```python
    if len(sys.argv) > 1 and sys.argv[1] == "kb":
        from claude_tap.prompt_kb.cli import kb_main

        sys.exit(kb_main(sys.argv[2:]))
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_cli.py -q`
Expected: 4 passed

- [ ] **Step 5: 手动验证 CLI 分发**

Run: `python3 -m claude_tap kb status`
Expected: 输出 `snapshots=0 chunks=0 pending=0 failed=0 indexed=0` + `embedder=none`（或已有数据的真实计数）

- [ ] **Step 6: Commit**

```bash
git add claude_tap/prompt_kb/cli.py claude_tap/cli.py tests/prompt_kb/test_cli.py
git commit -m "feat(prompt_kb): 新增 claude-tap kb 子命令"
```

---

### Task 8: dashboard API 路由 + 索引线程

**Files:**
- Modify: `claude_tap/live.py`（路由注册约 228-252 行 + 新增 4 个 handler + `start()` 内启动索引线程）
- Test: `tests/prompt_kb/test_kb_api.py`

**Interfaces:**
- Consumes: `KbStore`、`search`/`SnapshotResult`、`run_index_loop`、`ensure_embedder_meta`、`load_config`/`create_embedder`、`extract_unprocessed`、`timeline`
- Produces（dashboard.html 依赖的 JSON 契约）:
  - `GET /api/kb/search?q=&client=&kind=&limit=` → `200 {"results": [{"snapshot_id", "client", "model", "first_seen", "last_seen", "session_count", "hits": [{"kind", "title", "text", "score"}]}]}`；未装 rag 依赖 → `501 {"error": "rag_extra_missing", "hint": "..."}`；embedder 不匹配 → `409 {"error": "reindex_required", "hint": "..."}`
  - `GET /api/kb/status` → `200 {"available": bool, "stats": {...}, "embedder": str | null, "hint": str | null}`
  - `POST /api/kb/reindex` → `202 {"started": true}`（后台线程执行 rebuild）或 `501`
  - `GET /api/kb/timeline?client=&model=` → `200 {"versions": [{"id", "content_hash", "first_seen", "last_seen", "session_count"}]}`

- [ ] **Step 1: 写失败测试**

参照 `tests/test_dashboard.py:1068 test_dashboard_server_serves_session_api_and_exports` 的 server 启动模式（`LiveViewerServer(port=0, migrate_from=..., dashboard_mode=True)` + aiohttp client）：

```python
# tests/prompt_kb/test_kb_api.py
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_kb_api.py -q`
Expected: FAIL（404 on /api/kb/*）

- [ ] **Step 3: 实现**

`claude_tap/live.py`：

1. 文件头部 import 区加（live.py 当前**没有** `threading` 和 logger，一并补上）：

```python
import threading
import logging

from claude_tap.prompt_kb.embed import EmbedderUnavailable, create_embedder, load_config
from claude_tap.prompt_kb.index import ensure_embedder_meta, rebuild_index, run_index_loop
from claude_tap.prompt_kb.search import ReindexRequired, search as kb_search
from claude_tap.prompt_kb.store import KbStore

logger = logging.getLogger(__name__)
```

2. `start()` 路由注册区（`/api/stats` 之后）加：

```python
        app.router.add_get("/api/kb/search", self._handle_kb_search)
        app.router.add_get("/api/kb/status", self._handle_kb_status)
        app.router.add_post("/api/kb/reindex", self._handle_kb_reindex)
        app.router.add_get("/api/kb/timeline", self._handle_kb_timeline)
```

3. `start()` 末尾（`self.dashboard_mode` 为真时；该属性定义于 `live.py:207`）启动索引线程：

```python
        if self.dashboard_mode:
            self._kb_stop = threading.Event()
            self._kb_thread = threading.Thread(
                target=run_index_loop,
                kwargs={"stop_event": self._kb_stop},
                daemon=True,
                name="prompt-kb-indexer",
            )
            self._kb_thread.start()
```

`stop()` 方法开头加：

```python
        kb_stop = getattr(self, "_kb_stop", None)
        if kb_stop is not None:
            kb_stop.set()
```

4. 新增 handler（class 内，与 `_handle_stats` 同级）：

```python
    def _kb_embedder(self):
        """Process-wide lazy embedder shared by search and reindex."""
        if getattr(self, "_kb_embedder_instance", None) is None:
            embedder = create_embedder(load_config())
            store = KbStore.default()
            ensure_embedder_meta(store, embedder)
            self._kb_embedder_instance = embedder
        return self._kb_embedder_instance

    @staticmethod
    def _kb_unavailable_response(exc: Exception) -> web.Response:
        return web.json_response(
            {"error": "rag_extra_missing", "hint": str(exc)}, status=501,
        )

    async def _handle_kb_search(self, request: web.Request) -> web.Response:
        query = request.query.get("q", "").strip()
        if not query:
            return web.json_response({"results": []})
        try:
            embedder = self._kb_embedder()
        except EmbedderUnavailable as exc:
            return self._kb_unavailable_response(exc)
        try:
            results = kb_search(
                KbStore.default(), embedder, query,
                client=request.query.get("client") or None,
                kind=request.query.get("kind") or None,
                limit=int(request.query.get("limit", "10")),
            )
        except ReindexRequired as exc:
            return web.json_response(
                {"error": "reindex_required", "hint": str(exc)}, status=409,
            )
        return web.json_response({"results": [
            {
                "snapshot_id": group.snapshot_id,
                "client": group.client,
                "model": group.model,
                "first_seen": group.first_seen,
                "last_seen": group.last_seen,
                "session_count": group.session_count,
                "hits": [
                    {"kind": h.kind, "title": h.title, "text": h.text, "score": h.score}
                    for h in group.hits
                ],
            }
            for group in results
        ]})

    async def _handle_kb_status(self, request: web.Request) -> web.Response:
        store = KbStore.default()
        try:
            embedder = self._kb_embedder()
        except EmbedderUnavailable as exc:
            return web.json_response({
                "available": False, "stats": store.stats(),
                "embedder": None, "hint": str(exc),
            })
        return web.json_response({
            "available": True, "stats": store.stats(),
            "embedder": embedder.name, "hint": None,
        })

    async def _handle_kb_reindex(self, request: web.Request) -> web.Response:
        try:
            embedder = self._kb_embedder()
        except EmbedderUnavailable as exc:
            return self._kb_unavailable_response(exc)

        def _rebuild() -> None:
            try:
                rebuild_index(KbStore.default(), embedder)
            except Exception:  # noqa: BLE001
                logger.exception("kb reindex failed")

        threading.Thread(target=_rebuild, daemon=True, name="prompt-kb-reindex").start()
        return web.json_response({"started": True}, status=202)

    async def _handle_kb_timeline(self, request: web.Request) -> web.Response:
        store = KbStore.default()
        versions = store.timeline(
            request.query.get("client", ""), request.query.get("model", ""),
        )
        return web.json_response({"versions": [
            {
                "id": row["id"], "content_hash": row["content_hash"],
                "first_seen": row["first_seen"], "last_seen": row["last_seen"],
                "session_count": row["session_count"],
            }
            for row in versions
        ]})
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_kb_api.py -q`
Expected: 4 passed

- [ ] **Step 5: 回归**

Run: `python3 -m pytest tests/test_dashboard.py tests/test_live.py -q`
Expected: 全 pass（路由增加不影响既有行为）

- [ ] **Step 6: Commit**

```bash
git add claude_tap/live.py tests/prompt_kb/test_kb_api.py
git commit -m "feat(prompt_kb): dashboard 新增 /api/kb/* 路由与后台索引线程"
```

---

### Task 9: dashboard.html 知识库页面

**Files:**
- Modify: `claude_tap/dashboard.html`（view-toggle、新 section、JS、DASHBOARD_I18N）
- Test: `tests/prompt_kb/test_kb_page.py`

**Interfaces:**
- Consumes: Task 8 的 4 个 JSON API 契约；`DASHBOARD_I18N` / `t(key)` / `data-i18n` 机制
- Produces: 第三个视图 `data-view="kb"` + `#kb-view` section；`showKbView()`；`kbSearch()`；`kbLoadStatus()`

页面结构（中文为源语言）：
- 搜索框 `#kb-query`（placeholder「搜索 prompt 规则或工具定义…」）+ 过滤器 `#kb-kind`（全部/工具/prompt 段落）+ 搜索按钮
- 状态行 `#kb-status`（embedder 可用性、indexed/pending/failed 计数；不可用时显示安装提示 `pip install 'claude-tap[rag]'`）
- 结果列表 `#kb-results`：每组一张卡片（client / model / first seen / sessions + 最多 3 条命中，含 kind 徽章、title、score、text 片段）+「版本历史」展开按钮（拉 `/api/kb/timeline` 渲染版本列表）
- reindex 按钮（POST `/api/kb/reindex`）

- [ ] **Step 1: 写失败测试**

跟随 `tests/test_dashboard.py:705 test_dashboard_template_exposes_session_delete_controls` 的模板断言风格：

```python
# tests/prompt_kb/test_kb_page.py
from claude_tap.dashboard import read_dashboard_template


def test_kb_view_present_in_template():
    html = read_dashboard_template()
    assert 'data-view="kb"' in html
    assert 'id="kb-view"' in html
    assert 'id="kb-query"' in html
    assert 'id="kb-results"' in html
    assert 'id="kb-status"' in html


def test_kb_i18n_entries_zh_source():
    html = read_dashboard_template()
    assert '搜索 prompt 规则或工具定义' in html
    assert '"kb_view"' in html or "kb_view" in html
    # zh-CN entries exist alongside en fallbacks
    assert "Prompt 知识库" in html
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_kb_page.py -q`
Expected: FAIL（assertion errors）

- [ ] **Step 3: 实现（dashboard.html 四处修改）**

1. view-toggle（约 1002-1004 行）加第三个按钮：

```html
    <button class="view-toggle-btn" data-view="kb" role="tab" data-i18n="kb_view">Prompt 知识库</button>
```

2. `#stats-view` section 之后加：

```html
  <section id="kb-view" class="kb-view hidden" aria-label="Prompt knowledge base">
    <div class="kb-controls">
      <input id="kb-query" type="search" data-i18n-placeholder="kb_search_placeholder" placeholder="搜索 prompt 规则或工具定义…">
      <select id="kb-kind">
        <option value="" data-i18n="kb_kind_all">全部</option>
        <option value="tool" data-i18n="kb_kind_tool">工具</option>
        <option value="prompt_section" data-i18n="kb_kind_prompt">Prompt 段落</option>
      </select>
      <button id="kb-search-btn" data-i18n="kb_search_btn">搜索</button>
      <button id="kb-reindex-btn" data-i18n="kb_reindex_btn">重建索引</button>
    </div>
    <div id="kb-status" class="kb-status"></div>
    <div id="kb-results" class="kb-results"></div>
  </section>
```

配少量 CSS（沿用现有 var(--border)/var(--bg-card) 变量）：`.kb-view { display:flex; flex-direction:column; gap:12px; }`、`.kb-controls { display:flex; gap:8px; }`、`.kb-status { font-size:12px; color:var(--text-secondary); }`、`.kb-group { border:1px solid var(--border); border-radius:var(--radius-sm); padding:10px; background:var(--bg-card); }`、`.kb-hit { margin-top:8px; font-size:13px; }`、`.kb-hit-kind { font-size:11px; border:1px solid var(--border); border-radius:8px; padding:0 6px; margin-right:6px; }`。

3. JS：参照 `showStatsView()`/`updateViewToggle()` 模式加：

```javascript
function showKbView() {
  state.view = "kb";
  document.body.classList.remove("detail-route");
  $("#list-view").classList.add("hidden");
  $("#detail-view").classList.add("hidden");
  $("#stats-view").classList.add("hidden");
  $("#kb-view").classList.remove("hidden");
  updateViewToggle();
  kbLoadStatus().catch(console.error);
}
```

`updateViewToggle()` 中 active 判断改为 `btn.dataset.view === state.view`（list 为默认）。

```javascript
async function kbLoadStatus() {
  const resp = await fetch("/api/kb/status");
  const data = await resp.json();
  const el = $("#kb-status");
  if (!data.available) {
    el.textContent = t("kb_unavailable") + " " + (data.hint || "");
    return;
  }
  const s = data.stats;
  el.textContent = `${data.embedder} · indexed=${s.indexed} pending=${s.pending} failed=${s.failed} · snapshots=${s.snapshots}`;
}

async function kbSearch() {
  const query = $("#kb-query").value.trim();
  if (!query) return;
  const params = new URLSearchParams({ q: query });
  if ($("#kb-kind").value) params.set("kind", $("#kb-kind").value);
  const resp = await fetch(`/api/kb/search?${params}`);
  const data = await resp.json();
  if (resp.status === 501 || resp.status === 409) {
    $("#kb-results").textContent = data.hint || data.error;
    return;
  }
  renderKbResults(data.results || []);
}

function renderKbResults(results) {
  const container = $("#kb-results");
  container.innerHTML = "";
  if (!results.length) {
    container.textContent = t("kb_no_results");
    return;
  }
  for (const group of results) {
    const card = document.createElement("div");
    card.className = "kb-group";
    const header = document.createElement("div");
    header.textContent = `${group.client} / ${group.model} · ${t("kb_first_seen")} ${group.first_seen} · sessions ${group.session_count}`;
    card.appendChild(header);
    for (const hit of group.hits) {
      const hitEl = document.createElement("div");
      hitEl.className = "kb-hit";
      const badge = document.createElement("span");
      badge.className = "kb-hit-kind";
      badge.textContent = hit.kind === "tool" ? t("kb_kind_tool") : t("kb_kind_prompt");
      hitEl.appendChild(badge);
      hitEl.appendChild(document.createTextNode(
        `${hit.title} (${hit.score.toFixed(3)}) — ${hit.text.slice(0, 200)}`));
      card.appendChild(hitEl);
    }
    const timelineBtn = document.createElement("button");
    timelineBtn.textContent = t("kb_timeline");
    timelineBtn.addEventListener("click", () => kbLoadTimeline(group, card));
    card.appendChild(timelineBtn);
    container.appendChild(card);
  }
}

async function kbLoadTimeline(group, card) {
  const params = new URLSearchParams({ client: group.client, model: group.model });
  const resp = await fetch(`/api/kb/timeline?${params}`);
  const data = await resp.json();
  const list = document.createElement("ul");
  for (const v of data.versions || []) {
    const item = document.createElement("li");
    item.textContent = `${v.first_seen} · sessions ${v.session_count} · ${v.content_hash.slice(0, 8)}`;
    if (v.id === group.snapshot_id) item.style.fontWeight = "bold";
    list.appendChild(item);
  }
  card.appendChild(list);
}
```

事件绑定（与既有绑定同处）：`$("#kb-search-btn").addEventListener("click", () => kbSearch().catch(console.error));`、`$("#kb-query").addEventListener("keydown", e => { if (e.key === "Enter") kbSearch().catch(console.error); });`、`$("#kb-reindex-btn").addEventListener("click", async () => { await fetch("/api/kb/reindex", {method: "POST"}); await kbLoadStatus(); });`；view-toggle 的 click 分发处加 `if (btn.dataset.view === "kb") showKbView();`。

4. `DASHBOARD_I18N` 两个语言块都加条目（zh-CN 为源）：

```javascript
// zh-CN
kb_view: "Prompt 知识库",
kb_search_placeholder: "搜索 prompt 规则或工具定义…",
kb_kind_all: "全部", kb_kind_tool: "工具", kb_kind_prompt: "Prompt 段落",
kb_search_btn: "搜索", kb_reindex_btn: "重建索引",
kb_no_results: "没有命中的结果",
kb_first_seen: "首次出现",
kb_timeline: "版本历史",
kb_unavailable: "知识库不可用：",
// en
kb_view: "Prompt KB",
kb_search_placeholder: "Search prompt rules or tool definitions…",
kb_kind_all: "All", kb_kind_tool: "Tool", kb_kind_prompt: "Prompt section",
kb_search_btn: "Search", kb_reindex_btn: "Reindex",
kb_no_results: "No matching results",
kb_first_seen: "first seen",
kb_timeline: "Version history",
kb_unavailable: "Knowledge base unavailable: ",
```

（注意：`dashboard.html` 当前工作区可能有并行会话的未提交改动，编辑前先 `git status --short claude_tap/dashboard.html` 确认基线，锚点以文件实际内容为准。）

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest tests/prompt_kb/test_kb_page.py -q`
Expected: 2 passed

- [ ] **Step 5: 浏览器冒烟（有 rag 依赖时跳过即可，无依赖验证降级）**

Run: `python3 -m pytest tests/test_dashboard.py -q`
Expected: 全 pass（模板改动不破坏既有 dashboard 行为）

- [ ] **Step 6: Commit**

```bash
git add claude_tap/dashboard.html tests/prompt_kb/test_kb_page.py
git commit -m "feat(prompt_kb): dashboard 新增 Prompt 知识库页面"
```

---

### Task 10: 打包与文档

**Files:**
- Modify: `pyproject.toml:51-60`（optional-dependencies 加 rag）
- Modify: `README.md` / `README_zh.md`（Features/Install 附近加一小节）
- Modify: `claude_tap/prompt_kb/__init__.py`（公开接口 re-export）
- Test: `tests/prompt_kb/test_packaging.py`

**Interfaces:**
- Consumes: 全部前序任务
- Produces: `pip install 'claude-tap[rag]'` 可用；`from claude_tap.prompt_kb import KbStore, search, index_pending, rebuild_index, run_index_loop`

- [ ] **Step 1: 写失败测试**

```python
# tests/prompt_kb/test_packaging.py
import tomllib
from pathlib import Path


def test_rag_extra_declared():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "rag" in extras
    assert any(dep.startswith("sentence-transformers") for dep in extras["rag"])
    assert any(dep.startswith("numpy") for dep in extras["rag"])


def test_public_api_reexports():
    from claude_tap.prompt_kb import KbStore, index_pending, rebuild_index, run_index_loop, search

    assert callable(search) and callable(index_pending)
    assert callable(rebuild_index) and callable(run_index_loop)
    assert KbStore is not None
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest tests/prompt_kb/test_packaging.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

`pyproject.toml` 的 `[project.optional-dependencies]` 加：

```toml
rag = [
    "sentence-transformers>=2.7",
    "numpy>=1.26",
]
```

`claude_tap/prompt_kb/__init__.py`：

```python
"""Local prompt/tool knowledge base over captured trace snapshots."""

from claude_tap.prompt_kb.index import index_pending, rebuild_index, run_index_loop
from claude_tap.prompt_kb.search import search
from claude_tap.prompt_kb.store import KbStore

__all__ = ["KbStore", "index_pending", "rebuild_index", "run_index_loop", "search"]
```

`README_zh.md`「为什么用」列表后加一小节，`README.md` 对应英文：

```markdown
### Prompt 知识库（可选）

安装 `pip install 'claude-tap[rag]'` 后，dashboard 的「Prompt 知识库」页可对本地
采集到的各家 CLI system prompt 与工具定义做语义搜索，并查看每个 client/model 的
prompt 版本时间线。索引完全在本地完成（默认 `intfloat/multilingual-e5-small`，
可用 `CLAUDE_TAP_KB_EMBEDDER=api` 等环境变量切换到 OpenAI 兼容 embedding API）。
命令行等价物：`claude-tap kb search "..."` / `claude-tap kb reindex` / `claude-tap kb status`。
```

- [ ] **Step 4: 运行确认通过 + 全量回归**

Run: `python3 -m pytest tests/prompt_kb/ -q && python3 -m pytest tests/ -q`
Expected: 全 pass

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md README_zh.md claude_tap/prompt_kb/__init__.py tests/prompt_kb/test_packaging.py
git commit -m "feat(prompt_kb): 打包 rag 可选依赖并补充文档"
```

---

## Self-Review 记录

- **Spec 覆盖**：模块划分(T1-T7)、数据模型(T1)、切块规则(T2)、embedding 配置(T3)、数据流/懒索引/reindex(T4/T5/T8)、检索分组(T6)、CLI(T7)、API(T8)、页面(T9)、降级路径(T3/T6/T8/T9 的 501/提示)、验收标准 1-4 有对应任务与测试；验收标准 5（<200ms）为千级 chunk 下 numpy 暴力检索的自然结果，不单列测试
- **类型一致性**：`upsert_snapshot` 返回 `tuple[int, bool]` 在 T1/T4/T5/T6/T7/T8 测试中一致；`stats()` 五键在 T1/T5/T7/T8 一致；`ensure_embedder_meta` 在 T5 定义、T6/T7/T8 消费一致；`run_index_loop` 签名 T5 定义、T8 以 kwargs 调用一致；`create_session`/`get_trace_store`/`resolve_db_path`/`dashboard_mode` 均已对照当前源码核实（`trace_store.py:103`、`trace_store.py:73`、`trace_store.py:60`、`live.py:207`）
- **已知实现时核对点**：dashboard.html 可能有并行会话的未提交改动（T9 编辑前先看基线）
