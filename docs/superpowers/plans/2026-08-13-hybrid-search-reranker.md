# 知识库混合检索与重排（FTS5 hybrid + reranker）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 prompt 知识库加上关键词检索通道（FTS5 双路分词）与 cross-encoder 重排，使字面命中查询可召回、最终分数校准可比。

**Architecture:** 三通道召回（向量余弦 + trigram FTS BM25 + jieba FTS BM25）→ RRF(k=60) 融合 → 跨快照去重 → bge-reranker-base 重排（sigmoid 校准）→ 现有分组/截断收尾。新模块 `tokenize.py`（jieba 预切分）与 `rerank.py`（Reranker 协议 + LocalReranker）分离职责；store.py 管 4 张 contentless FTS 表的同步与回填；search.py 只做管线编排，全链路优雅降级。

**Tech Stack:** SQLite FTS5（trigram/unicode61，contentless）、jieba（纯 Python 中文分词）、sentence-transformers CrossEncoder（BAAI/bge-reranker-base）、numpy、pytest。

**Spec:** `docs/superpowers/specs/2026-08-13-hybrid-search-reranker-design.md`

## Global Constraints

- 测试一律用 `.venv/bin/pytest`（**禁止** `uv run`；DLP 拦截外网，`.venv` 无 pip 直链时需 `-i https://mirrors.aliyun.com/pypi/simple/`）
- Commit message 一律中文（保留英文 `type(scope):` 前缀）；代码与注释用英文
- 每个新 Python 文件首行后接 `from __future__ import annotations`
- ruff：line-length 120，select E/F/W/I
- `mcp` 依赖钉 `<2`，不得改动
- `jieba` 只进 `rag` 与 `dev` extra，**不进主依赖**；必须懒加载（模块顶部不得 `import jieba`）
- 对外返回结构向后兼容：只加字段（`reranked`），不改既有字段语义；`search()`/`search_messages()` 的返回值从 `list` 变为 `tuple[list, bool]` 是本次唯一签名破坏，所有调用方在 Task 5 内同步适配
- KB 为本地功能，不得新增外发路径；reranker 模型下载走现有 HF/modelscope 缓存
- 已验证的环境事实（写代码时直接采信，不要重新试验）：
  - bm25() 返回负值，**越小（越负）越好**；`ORDER BY bm25(t)` 升序即最佳在前，取负得"越大越好"
  - contentless 表删除：`INSERT INTO t (t, rowid, text) VALUES ('delete', ?, ?)`；清空：`INSERT INTO t (t) VALUES ('delete-all')`
  - trigram 分词器折叠大小写；`jieba.cut_for_search("取消定时任务cron")` == `"取消 定时 任务 cron"`
  - `.venv` 已装 jieba 0.42.1 与 numpy；**未装** sentence-transformers（reranker 降级路径是测试默认）

---

### Task 1: tokenize 模块 + jieba 依赖

**Files:**
- Create: `claude_tap/prompt_kb/tokenize.py`
- Test: `tests/prompt_kb/test_tokenize.py`
- Modify: `pyproject.toml:52-67`（dev 与 rag extra）

**Interfaces:**
- Produces: `segment(text: str) -> str`——jieba `cut_for_search` 切词、空格连接；jieba 缺失时返回原文并发一次性 `UserWarning`。store.py 写 FTS（Task 2）与 search.py 构造 MATCH（Task 5）都必须用这个函数。

- [ ] **Step 1: Write the failing test**

```python
"""tokenize.segment(): jieba cut_for_search with graceful degradation."""

from __future__ import annotations

import builtins
import warnings

import pytest

from claude_tap.prompt_kb import tokenize


@pytest.fixture(autouse=True)
def _reset_jieba_cache(monkeypatch):
    monkeypatch.setattr(tokenize, "_jieba", None)
    monkeypatch.setattr(tokenize, "_jieba_failed", False)


def test_segment_exact_verified_output():
    assert tokenize.segment("取消定时任务cron") == "取消 定时 任务 cron"


def test_segment_mixed_keeps_english_tokens():
    tokens = tokenize.segment("用 CronDelete 取消定时任务").split()
    assert "CronDelete" in tokens
    assert "定时" in tokens


def test_segment_degrades_without_jieba(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jieba":
            raise ImportError("no jieba")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.warns(UserWarning, match="jieba"):
        assert tokenize.segment("取消定时任务") == "取消定时任务"
    # Second call: degradation is permanent for the process, no repeated warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tokenize.segment("再次调用") == "再次调用"


def test_module_top_does_not_import_jieba():
    source = open(__import__("claude_tap.prompt_kb.tokenize", fromlist=["x"]).__file__).read()
    assert "\nimport jieba" not in source
    assert "\nimport jieba." not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/prompt_kb/test_tokenize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claude_tap.prompt_kb.tokenize'`

- [ ] **Step 3: Write the implementation**

`claude_tap/prompt_kb/tokenize.py`:

```python
"""jieba-based segmentation for FTS keyword search over zh+en mixed text.

Both the FTS write path (store.py) and the query path (search.py) MUST use
segment() so index terms and query terms are cut identically.
"""

from __future__ import annotations

import warnings

_jieba = None
_jieba_failed = False


def _load_jieba():
    global _jieba, _jieba_failed
    if _jieba is not None or _jieba_failed:
        return _jieba
    try:
        import jieba
    except ImportError:
        _jieba_failed = True
        warnings.warn(
            "jieba is not installed; Chinese keyword search is degraded: pip install 'claude-tap[rag]'",
            stacklevel=2,
        )
        return None
    _jieba = jieba
    return _jieba


def segment(text: str) -> str:
    """Cut text into space-joined search tokens; raw text when jieba is missing."""
    jieba = _load_jieba()
    if jieba is None:
        return text
    return " ".join(jieba.cut_for_search(text))
```

`pyproject.toml` 两处（dev extra 里 jieba 供测试环境，rag extra 供用户）：

```toml
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-timeout>=2.3",
    "pexpect>=4.9",
    "coverage>=7.6",
    "matplotlib==3.11.1",
    "ruff>=0.11",
    "jieba>=0.42",
]
rag = [
    "sentence-transformers>=2.7",
    "numpy>=1.26",
    "jieba>=0.42",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/prompt_kb/test_tokenize.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/tokenize.py tests/prompt_kb/test_tokenize.py pyproject.toml
git commit -m "feat(prompt_kb): 新增 jieba 预切分模块——FTS 中文关键词检索的分词基础

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: store FTS 表 + 写路径同步 + fts_rank

**Files:**
- Modify: `claude_tap/prompt_kb/store.py`（SCHEMA、replace_chunks、upsert_message、delete_messages_for_session、新增 fts_rank）
- Test: `tests/prompt_kb/test_store.py`（追加）、`tests/prompt_kb/test_store_messages.py`（追加）

**Interfaces:**
- Consumes: `segment()`（Task 1）
- Produces: `KbStore.fts_rank(entity: str, tokenizer: str, match_query: str, limit: int) -> list[tuple[int, float]]`——entity ∈ {"chunks","messages"}，tokenizer ∈ {"tri","jieba"}，返回 `(rowid, 正向分数)` 最佳在前；FTS 表不存在或 MATCH 异常返回 `[]`。search.py（Task 5）消费。模块级常量 `FTS_ENTITIES = ("chunks", "messages")` 与函数 `fts_tables(entity) -> tuple[str, str]`。

- [ ] **Step 1: Write the failing test**

`tests/prompt_kb/test_store.py` 追加（文件已有 `_upsert(store, ...)` 辅助函数，直接复用）：

```python
def test_fts_synced_on_replace_chunks(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("tool", "shell", "CronDelete cancels scheduled cron jobs")])
    ranked = store.fts_rank("chunks", "tri", "CronDelete", 10)
    assert len(ranked) == 1 and ranked[0][0] == 1
    # jieba table got the segmented copy of the same text.
    ranked_jieba = store.fts_rank("chunks", "jieba", "CronDelete", 10)
    assert len(ranked_jieba) == 1 and ranked_jieba[0][0] == 1


def test_fts_chinese_word_match_via_jieba_channel(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("prompt_section", "指南", "取消定时任务的正确方法")])
    # 2-char Chinese word: trigram cannot match it, the jieba channel can.
    assert store.fts_rank("chunks", "tri", "定时", 10) == []
    ranked = store.fts_rank("chunks", "jieba", "定时", 10)
    assert len(ranked) == 1


def test_fts_updated_on_replace_chunks(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("tool", "shell", "alpha bravo charlie")])
    store.replace_chunks(snap_id, [("tool", "shell", "delta echo foxtrot")])
    assert store.fts_rank("chunks", "tri", "alpha", 10) == []
    assert len(store.fts_rank("chunks", "tri", "delta", 10)) == 1


def test_fts_rank_rejects_unknown_channel(trace_db):
    store = KbStore.default()
    import pytest

    with pytest.raises(ValueError):
        store.fts_rank("chunks", "bogus", "x", 10)
    with pytest.raises(ValueError):
        store.fts_rank("bogus", "tri", "x", 10)
```

`tests/prompt_kb/test_store_messages.py` 追加（复用该文件已有的 message upsert 辅助/参数风格）：

```python
def test_fts_synced_on_message_insert_and_session_delete(trace_db):
    store = KbStore.default()
    store.upsert_message(
        session_id="s1",
        record_index=0,
        message_index=0,
        client="codex",
        model="gpt-5",
        timestamp="2026-08-01T00:00:00Z",
        content_hash="m1",
        text="how to cancel a scheduled cron job",
        seen_at="t",
    )
    assert len(store.fts_rank("messages", "tri", "cron", 10)) == 1
    store.delete_messages_for_session("s1")
    assert store.fts_rank("messages", "tri", "cron", 10) == []


def test_fts_not_written_on_dedup_hit(trace_db):
    store = KbStore.default()
    kwargs = dict(
        session_id="s1",
        record_index=0,
        message_index=0,
        client="codex",
        model="gpt-5",
        timestamp="2026-08-01T00:00:00Z",
        content_hash="m1",
        text="unique phrase about reticulating splines",
        seen_at="t",
    )
    store.upsert_message(**kwargs)
    store.upsert_message(**{**kwargs, "session_id": "s2", "seen_at": "t2"})  # dedup hit
    ranked = store.fts_rank("messages", "tri", "reticulating", 10)
    assert len(ranked) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/prompt_kb/test_store.py -k fts tests/prompt_kb/test_store_messages.py -k fts -v`
Expected: FAIL with `AttributeError: 'KbStore' object has no attribute 'fts_rank'`

- [ ] **Step 3: Write the implementation**

`store.py` 顶部 import 区加：

```python
from claude_tap.prompt_kb.tokenize import segment
```

SCHEMA 常量末尾（`idx_kb_messages_dedup` 之后）追加：

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_chunks_tri USING fts5(text, content='', tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_chunks_jieba USING fts5(text, content='', tokenize='unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_messages_tri USING fts5(text, content='', tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts_messages_jieba USING fts5(text, content='', tokenize='unicode61');
```

模块级（SCHEMA 之后、`default_db_path` 之前）加：

```python
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
```

`replace_chunks()` 改为（整个方法替换）：

```python
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
```

`upsert_message()` 的插入分支末尾（`return int(cur.lastrowid), True` 之前）加一行：

```python
            _fts_insert(conn, "messages", int(cur.lastrowid), text)
```

`delete_messages_for_session()` 改为：

```python
    def delete_messages_for_session(self, session_id: str) -> int:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, text FROM kb_messages WHERE session_id=?", (session_id,)).fetchall()
            for row in rows:
                _fts_delete(conn, "messages", row["id"], row["text"])
            cur = conn.execute("DELETE FROM kb_messages WHERE session_id=?", (session_id,))
            return cur.rowcount
```

`KbStore` 新增方法（放在 `indexed_messages()` 之后）：

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/prompt_kb/test_store.py tests/prompt_kb/test_store_messages.py -v`
Expected: all passed（含新增 6 个 fts 用例；既有用例不受影响）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/store.py tests/prompt_kb/test_store.py tests/prompt_kb/test_store_messages.py
git commit -m "feat(prompt_kb): FTS5 双路分词表与写路径同步——trigram 子串 + jieba 中文词

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: FTS 迁移回填 + rebuild_fts

**Files:**
- Modify: `claude_tap/prompt_kb/store.py`（`_migrate`、新增 `rebuild_fts`）
- Test: `tests/prompt_kb/test_store.py`（追加迁移测试）

**Interfaces:**
- Consumes: `_fts_insert` / `fts_tables` / `_FTS_TABLE_BY_ENTITY`（Task 2）
- Produces: `KbStore.rebuild_fts() -> int`（重建行数）。CLI `kb rebuild-fts`（Task 7）消费。

- [ ] **Step 1: Write the failing test**

`tests/prompt_kb/test_store.py` 追加：

```python
def _build_legacy_db_without_fts(path):
    """A pre-hybrid-schema DB: main tables with data, no FTS tables, no meta flag."""
    import sqlite3 as _sq

    conn = _sq.connect(path)
    conn.execute(
        "CREATE TABLE kb_chunks (id INTEGER PRIMARY KEY, snapshot_id INTEGER NOT NULL,"
        " kind TEXT NOT NULL, title TEXT, text TEXT NOT NULL, embedding BLOB,"
        " index_state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("INSERT INTO kb_chunks (snapshot_id, kind, title, text) VALUES (1, 'tool', 'shell', 'legacy cron tooling')")
    conn.execute(
        "CREATE TABLE kb_messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,"
        " record_index INTEGER NOT NULL, message_index INTEGER NOT NULL, client TEXT NOT NULL,"
        " model TEXT NOT NULL, timestamp TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,"
        " last_seen TEXT NOT NULL, embedding BLOB, index_state TEXT NOT NULL DEFAULT 'pending',"
        " attempts INTEGER NOT NULL DEFAULT 0, role TEXT NOT NULL DEFAULT 'user')"
    )
    conn.execute(
        "INSERT INTO kb_messages (session_id, record_index, message_index, client, model,"
        " timestamp, content_hash, text, last_seen) VALUES ('s1', 0, 0, 'c', 'm', 't', 'h', 'legacy message text', 't')"
    )
    conn.execute("CREATE TABLE kb_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()


def test_fts_backfilled_on_open_for_legacy_db(trace_db, tmp_path):
    path = tmp_path / "legacy_kb.sqlite3"
    _build_legacy_db_without_fts(path)
    store = KbStore(path)  # migration runs on open
    assert len(store.fts_rank("chunks", "tri", "legacy", 10)) == 1
    assert len(store.fts_rank("messages", "tri", "legacy", 10)) == 1
    assert store.get_meta("fts_backfilled") == "1"
    store2 = KbStore(path)  # second open: no duplicate FTS rows
    assert len(store2.fts_rank("chunks", "tri", "legacy", 10)) == 1


def test_rebuild_fts_clears_and_reindexes(trace_db):
    store = KbStore.default()
    snap_id, _ = _upsert(store)
    store.replace_chunks(snap_id, [("tool", "shell", "rebuild me please")])
    assert store.rebuild_fts() == 1
    assert len(store.fts_rank("chunks", "tri", "rebuild", 10)) == 1
    assert store.rebuild_fts() == 1  # idempotent, no duplicates
    assert len(store.fts_rank("chunks", "tri", "rebuild", 10)) == 1
```

注：`trace_db` fixture 把 KB 库路径重定向到临时目录；`tmp_path` 用于独立构造 legacy 库。

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/prompt_kb/test_store.py -k "fts_backfilled or rebuild_fts" -v`
Expected: FAIL（`KbStore.rebuild_fts` 不存在 / legacy 库 fts_rank 返回空）

- [ ] **Step 3: Write the implementation**

`_migrate()` 末尾（boilerplate purge 块之后）追加：

```python
        fts_done = conn.execute("SELECT value FROM kb_meta WHERE key='fts_backfilled'").fetchone()
        if fts_done is None or fts_done["value"] != "1":
            self._backfill_fts(conn)
            conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('fts_backfilled', '1')")
```

`_migrate` 是 `@staticmethod`；在类中新增两个方法（放在 `fts_rank` 之后）：

```python
    @staticmethod
    def _backfill_fts(conn: sqlite3.Connection) -> None:
        """Full-scan FTS backfill for databases created before hybrid search."""
        for entity in FTS_ENTITIES:
            rows = conn.execute(f"SELECT id, text FROM {_FTS_TABLE_BY_ENTITY[entity]}").fetchall()
            for row in rows:
                _fts_insert(conn, entity, row["id"], row["text"])

    def rebuild_fts(self) -> int:
        """Clear and rebuild every FTS table from the main tables. Idempotent;
        use after installing jieba to upgrade pre-jieba segmented rows."""
        with self._connect() as conn:
            total = 0
            for entity in FTS_ENTITIES:
                for table in fts_tables(entity):
                    conn.execute(f"INSERT INTO {table} ({table}) VALUES ('delete-all')")
                rows = conn.execute(f"SELECT id, text FROM {_FTS_TABLE_BY_ENTITY[entity]}").fetchall()
                for row in rows:
                    _fts_insert(conn, entity, row["id"], row["text"])
                total += len(rows)
            conn.execute("INSERT OR REPLACE INTO kb_meta (key, value) VALUES ('fts_backfilled', '1')")
            return total
```

注意：`_migrate` 是静态方法且拿不到 `self`，所以 `_backfill_fts` 也做成静态方法接收 `conn`。

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/prompt_kb/test_store.py tests/prompt_kb/test_store_messages.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/store.py tests/prompt_kb/test_store.py
git commit -m "feat(prompt_kb): FTS 迁移回填与 rebuild_fts——存量库打开即索引，幂等可重建

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: reranker 模块 + 配置扩展 + FakeReranker

**Files:**
- Create: `claude_tap/prompt_kb/rerank.py`
- Modify: `claude_tap/prompt_kb/embed.py`（KbConfig 两个字段 + load_config 环境变量键 + 模型常量）
- Create: `tests/prompt_kb/fake_reranker.py`
- Test: `tests/prompt_kb/test_rerank.py`

**Interfaces:**
- Produces:
  - `RerankerUnavailable(Exception)`
  - `Reranker` 协议：`name: str`、`rerank(query: str, texts: list[str]) -> list[float]`（0–1 sigmoid 分）
  - `LocalReranker(model_name: str = DEFAULT_RERANKER_MODEL)`
  - `create_reranker(config: KbConfig) -> Reranker | None`（`reranker="off"` 时 None；加载失败抛 `RerankerUnavailable`）
  - `reranker_status(config: KbConfig) -> str`：`"off"` / `"unavailable"` / 模型名
  - `KbConfig` 新字段 `reranker: str = "on"`、`reranker_model: str = DEFAULT_RERANKER_MODEL`；环境变量 `CLAUDE_TAP_KB_RERANKER` / `CLAUDE_TAP_KB_RERANKER_MODEL`
  - `DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"`（定义在 embed.py，rerank.py import）

- [ ] **Step 1: Write the failing test**

`tests/prompt_kb/test_rerank.py`:

```python
"""LocalReranker sigmoid scoring, degradation contract, and config wiring."""

from __future__ import annotations

import builtins
import math
from pathlib import Path

import pytest

from claude_tap.prompt_kb.embed import DEFAULT_RERANKER_MODEL, KbConfig, load_config
from claude_tap.prompt_kb.rerank import (
    LocalReranker,
    RerankerUnavailable,
    create_reranker,
    reranker_status,
)
from tests.prompt_kb.fake_reranker import FakeReranker


class _StubModel:
    def __init__(self, logits):
        self._logits = logits

    def predict(self, pairs):
        assert all(len(pair) == 2 for pair in pairs)
        return self._logits[: len(pairs)]


def _stub_reranker(logits):
    reranker = LocalReranker.__new__(LocalReranker)
    reranker._model = _StubModel(logits)
    return reranker


def test_rerank_sigmoid_normalizes():
    scores = _stub_reranker([0.0, 2.0, -2.0]).rerank("q", ["a", "b", "c"])
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert scores[2] == pytest.approx(1.0 / (1.0 + math.exp(2.0)))
    assert all(0.0 < score < 1.0 for score in scores)


def test_rerank_empty_texts():
    assert _stub_reranker([]).rerank("q", []) == []


def test_load_failure_raises_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("no sentence-transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RerankerUnavailable, match="sentence-transformers"):
        LocalReranker()


def test_create_reranker_off_returns_none():
    assert create_reranker(KbConfig(reranker="off")) is None


def test_reranker_status_states(monkeypatch):
    assert reranker_status(KbConfig(reranker="off")) == "off"
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert reranker_status(KbConfig()) == "unavailable"


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("CLAUDE_TAP_KB_RERANKER", "off")
    monkeypatch.setenv("CLAUDE_TAP_KB_RERANKER_MODEL", "org/custom-reranker")
    config = load_config(path=Path("/nonexistent-kb-config.toml"))
    assert config.reranker == "off"
    assert config.reranker_model == "org/custom-reranker"


def test_config_defaults():
    config = load_config(path=Path("/nonexistent-kb-config.toml"))
    assert config.reranker == "on"
    assert config.reranker_model == DEFAULT_RERANKER_MODEL == "BAAI/bge-reranker-base"


def test_fake_reranker_deterministic_contract():
    scores = FakeReranker().rerank("alpha beta", ["alpha beta gamma", "nothing here", "alpha"])
    assert scores == [1.0, 0.0, 0.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/prompt_kb/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claude_tap.prompt_kb.rerank'`

- [ ] **Step 3: Write the implementation**

`embed.py`：`DEFAULT_LOCAL_MODEL` 常量后加：

```python
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
```

`KbConfig` 加两个字段（放在 `passage_prefix` 之后）：

```python
    reranker: str = "on"  # "on" | "off"
    reranker_model: str = DEFAULT_RERANKER_MODEL
```

`load_config()` 的环境变量键元组加 `"reranker"` 与 `"reranker_model"`。

`claude_tap/prompt_kb/rerank.py`:

```python
"""Cross-encoder reranker: rescore fused candidates for calibrated ordering.

The reranker is an enhancement, never a requirement: every load/runtime
failure surfaces as RerankerUnavailable so callers can fall back to the
fused ranking (search reports reranked=false in that case).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from claude_tap.prompt_kb.embed import DEFAULT_RERANKER_MODEL

if TYPE_CHECKING:
    from claude_tap.prompt_kb.embed import KbConfig


class RerankerUnavailable(Exception):
    """The reranker model cannot be loaded."""


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class LocalReranker:
    """Local bge cross-encoder; predict() logits squashed through sigmoid."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerUnavailable(
                "sentence-transformers is not installed; install the optional dependency: pip install 'claude-tap[rag]'"
            ) from exc
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:
            # Download/load failures (network, TLS, disk, corrupt cache) must
            # degrade to the fused ranking, not crash the search path.
            raise RerankerUnavailable(f"failed to load reranker model {model_name!r}: {exc}") from exc
        self.name = f"reranker:{model_name}"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = self._model.predict([[query, text] for text in texts])
        return [1.0 / (1.0 + math.exp(-float(logit))) for logit in logits]


def create_reranker(config: KbConfig) -> Reranker | None:
    """None when configured off; raises RerankerUnavailable on load failure."""
    if config.reranker == "off":
        return None
    return LocalReranker(config.reranker_model)


def reranker_status(config: KbConfig) -> str:
    """Human-readable state for kb_status/stats: off | unavailable | model name."""
    if config.reranker == "off":
        return "off"
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "unavailable"
    return config.reranker_model
```

`tests/prompt_kb/fake_reranker.py`:

```python
"""Deterministic lexical-overlap reranker for tests (no model download)."""

from __future__ import annotations

import re


class FakeReranker:
    name = "fake-reranker"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return [0.0] * len(texts)
        return [
            len(query_tokens & set(re.findall(r"[a-z0-9]+", text.lower()))) / len(query_tokens) for text in texts
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/prompt_kb/test_rerank.py tests/prompt_kb/test_embed.py -v`
Expected: all passed（`sentence-transformers` 未装时 `reranker_status` 默认状态测试走 monkeypatch 路径，不依赖真实包）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/rerank.py claude_tap/prompt_kb/embed.py tests/prompt_kb/fake_reranker.py tests/prompt_kb/test_rerank.py
git commit -m "feat(prompt_kb): 新增 reranker 模块——bge-reranker-base 校准重排与三级降级契约

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: search.py 混合管线 + 全部调用方适配

**Files:**
- Modify: `claude_tap/prompt_kb/search.py`（整体重写为混合管线）
- Modify: `claude_tap/prompt_kb/mcp_server.py`（_ctx 三元组、reranked 字段、docstring）
- Modify: `claude_tap/live.py`（_kb_reranker、reranked 透传、rel_delta 参数）
- Modify: `claude_tap/prompt_kb/cli.py`（调用方解包 + reranked 行）
- Test: `tests/prompt_kb/test_search.py`、`tests/prompt_kb/test_search_messages.py`、`tests/prompt_kb/test_mcp_server.py`、`tests/prompt_kb/test_kb_api.py`、`tests/prompt_kb/test_cli.py`（既有用例机械适配 + 新增管线用例）

**Interfaces:**
- Consumes: `fts_rank`（Task 2）、`segment`（Task 1）、`Reranker` 协议（Task 4）、`FakeReranker`
- Produces（签名变更，全部调用方本任务内适配）：
  - `search(store, embedder, query, *, client=None, kind=None, limit=10, min_score=0.0, rel_delta=0.05, recall=20, reranker=None) -> tuple[list[SnapshotResult], bool]`
  - `search_messages(store, embedder, query, *, client=None, limit=10, min_score=0.0, rel_delta=0.05, recall=20, reranker=None) -> tuple[list[SessionResult], bool]`
  - 语义：`reranked=True` 时 hit.score 为重排校准分，`min_score` 严格 `>` 过滤，`rel_delta` 忽略；`reranked=False` 时 hit.score 为向量余弦回退分（FTS 独有候选可为 0.0），`min_score` 用 `>=` 过滤，`rel_delta` 维持旧行为

- [ ] **Step 1: Write the failing tests**

`tests/prompt_kb/test_search.py` **既有用例机械适配**：所有 `results = search(...)` 改为 `results, _ = search(...)`；所有 `search(...) == []` 改为 `search(...)[0] == []`；`all_hits = search(...)` / `filtered = search(...)` / `everything = search(...)` 同样解包第一项。然后追加：

```python
from claude_tap.prompt_kb.search import _match_query, _rrf_fuse  # noqa: E402
from tests.prompt_kb.fake_reranker import FakeReranker  # noqa: E402


def test_match_query_sanitizes():
    assert _match_query('怎么取消 "cron" (定时任务)?') == "怎么取消 OR cron OR 定时任务"
    assert _match_query("!!!") == ""


def test_rrf_fusion_prefers_multi_channel_hits():
    fused = _rrf_fuse([[(1, 0.9), (2, 0.8)], [(2, 5.0), (3, 4.0)]], 10)
    assert fused[0] == 2  # present in both channels
    assert set(fused) == {1, 2, 3}


def _seed_chinese(store: KbStore) -> None:
    sid, _ = store.upsert_snapshot(
        content_hash="h-zh", client="codex", provider="p", model="m",
        system_prompt="s", developer_prompt="", tools_json="[]", seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(sid, [("prompt_section", "指南", "取消定时任务的正确方法")])


def test_jieba_channel_recalls_chinese_vector_miss(trace_db):
    store = KbStore.default()
    _seed_chinese(store)
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    # FakeEmbedder has zero token overlap for Chinese; only the FTS channels can hit.
    results, _ = search(store, embedder, "定时任务")
    assert len(results) == 1
    assert results[0].hits[0].title == "指南"


def test_trigram_channel_recalls_substring_vector_miss(trace_db):
    store = KbStore.default()
    sid, _ = store.upsert_snapshot(
        content_hash="h-cron", client="codex", provider="p", model="m",
        system_prompt="s", developer_prompt="", tools_json="[]", seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(sid, [("tool", "cron", "CronDelete cancels scheduled cron jobs")])
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    # "CronDele" is not a whole token: zero FakeEmbedder overlap, trigram substring hit.
    results, _ = search(store, embedder, "CronDele")
    assert len(results) == 1
    assert results[0].hits[0].title == "cron"


def test_reranker_replaces_scores_and_drops_irrelevant(trace_db):
    store = _indexed_store()
    results, reranked = search(store, FakeEmbedder(), "shell sandbox", reranker=FakeReranker())
    assert reranked is True
    # FakeReranker: full query-token overlap on the shell chunk (1.0);
    # the prose chunk scores 0.0 and is dropped by the strict reranked filter.
    assert len(results) == 1
    assert results[0].hits[0].title == "shell"
    assert results[0].hits[0].score == pytest.approx(1.0)


def test_reranker_runtime_failure_falls_back(trace_db):
    store = _indexed_store()

    class BrokenReranker:
        name = "broken"

        def rerank(self, query, texts):
            raise RuntimeError("boom")

    results, reranked = search(store, FakeEmbedder(), "shell sandbox", reranker=BrokenReranker())
    assert reranked is False
    assert results[0].hits[0].title == "shell"  # cosine fallback still ranks


def test_rel_delta_ignored_when_reranked(trace_db):
    store = KbStore.default()
    _seed_overlap(store)
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    results, reranked = search(store, embedder, "alpha beta gamma delta epsilon zeta", reranker=FakeReranker())
    assert reranked is True
    assert len(results) == 1  # strict reranked filter, no rel_delta involved
```

`tests/prompt_kb/test_search_messages.py`：既有用例机械适配（同上解包），追加：

```python
from tests.prompt_kb.fake_reranker import FakeReranker  # noqa: E402


def test_messages_reranked_flag(trace_db, monkeypatch):
    # reuse the file's existing seeding helper/fixture for one indexed message
    ...
```

**注意**：test_search_messages.py 的现有 fixture 名以文件实际内容为准（实现者读文件适配，模式同 test_search.py：`results, reranked = search_messages(store, embedder, "...", reranker=FakeReranker())`，断言 `reranked is True` 且命中仍存在；再有一个无 reranker 的 `reranked is False` 断言）。

`tests/prompt_kb/test_mcp_server.py` 适配：

- `ctx` fixture 中 `monkeypatch.setattr(mcp_server, "_get_ctx", lambda: (store, embedder))` 改为 `lambda: (store, embedder, None)`
- `test_kb_search_returns_chunks_section` 的 `assert set(result) == {"chunks", "messages"}` 改为 `assert set(result) == {"chunks", "messages", "reranked"}`
- 追加：

```python
def test_kb_search_reranked_flags(ctx, monkeypatch):
    store, _ = ctx
    _seed(store)
    result = mcp_server.kb_search("shell sandbox")
    assert result["reranked"] is False  # ctx fixture has reranker=None

    from tests.prompt_kb.fake_reranker import FakeReranker

    monkeypatch.setattr(mcp_server, "_get_ctx", lambda: (store, FakeEmbedder(), FakeReranker()))
    result = mcp_server.kb_search("shell sandbox")
    assert result["reranked"] is True
```

`tests/prompt_kb/test_cli.py` 适配：`seeded_kb` fixture 加一行 `monkeypatch.setattr("claude_tap.prompt_kb.cli.create_reranker", lambda config: None)`；追加：

```python
def test_kb_search_prints_reranked_line(seeded_kb, capsys):
    assert kb_main(["search", "shell sandbox"]) == 0
    assert "reranked: no" in capsys.readouterr().out
```

`tests/prompt_kb/test_kb_api.py` 适配：构造 server 的 fixture 加 `monkeypatch.setattr("claude_tap.live.create_reranker", lambda config: None)`；`test_kb_search_route` 追加断言 `payload["reranked"] is False`。

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/prompt_kb/test_search.py tests/prompt_kb/test_search_messages.py tests/prompt_kb/test_mcp_server.py tests/prompt_kb/test_cli.py tests/prompt_kb/test_kb_api.py -x -q`
Expected: FAIL（`_match_query` 不存在 / tuple 解包 TypeError）

- [ ] **Step 3: Write the implementation**

`search.py` 顶部 import 区改为：

```python
"""Hybrid search: vector + FTS keyword channels, RRF-fused, reranked."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from claude_tap.prompt_kb.embed import Embedder, EmbedderUnavailable
from claude_tap.prompt_kb.rerank import Reranker
from claude_tap.prompt_kb.store import KbStore
from claude_tap.prompt_kb.tokenize import segment
```

保留：`ReindexRequired`、`SearchHit`、`SnapshotResult`、`MessageHit`、`SessionResult`、`_cosine_scores`、`_check_embedder_meta`、`_dedup_across_snapshots`（全部原样不动）。新增模块级：

```python
_RRF_K = 60
_WORD_RE = re.compile(r"[A-Za-z0-9_一-鿿]+")


def _match_query(text: str) -> str:
    """Sanitize raw text into an FTS5 MATCH query (OR of word tokens)."""
    return " OR ".join(_WORD_RE.findall(text))


def _rrf_fuse(rankings: list[list[tuple[int, float]]], limit: int) -> list[int]:
    """Reciprocal-rank fusion over (rowid, score) rankings → fused rowids, best first."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (rowid, _score) in enumerate(ranking, 1):
            fused[rowid] = fused.get(rowid, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(fused, key=fused.__getitem__, reverse=True)[:limit]


def _fts_channels(store: KbStore, entity: str, query: str, recall: int) -> list[list[tuple[int, float]]]:
    """Keyword rankings: trigram on raw terms, jieba-table on segmented terms."""
    channels = []
    tri_match = _match_query(query)
    if tri_match:
        channels.append(store.fts_rank(entity, "tri", tri_match, recall))
    jieba_match = _match_query(segment(query))
    if jieba_match:
        channels.append(store.fts_rank(entity, "jieba", jieba_match, recall))
    return channels


def _final_scores(
    reranker: Reranker | None, query: str, texts: list[str], fallback: list[float]
) -> tuple[list[float], bool]:
    """Reranker (sigmoid-calibrated) scores when available, else cosine fallback.

    A reranker that raises at runtime degrades to the fallback: search must
    never fail because an enhancement failed.
    """
    if reranker is None:
        return fallback, False
    if not texts:
        return fallback, True
    try:
        return [float(score) for score in reranker.rerank(query, texts)], True
    except Exception:  # noqa: BLE001 - a broken reranker must not break search
        return fallback, False
```

`search()` 整体替换为：

```python
def search(
    store: KbStore,
    embedder: Embedder,
    query: str,
    *,
    client: str | None = None,
    kind: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    rel_delta: float = 0.05,
    recall: int = 20,
    reranker: Reranker | None = None,
) -> tuple[list[SnapshotResult], bool]:
    """Hybrid search over chunks; returns (groups, reranked).

    Three channels (vector cosine, trigram FTS, jieba FTS) are RRF-fused into
    candidates; the reranker rescores them when available. Scores are reranker
    scores when reranked=True (calibrated, rel_delta ignored), else cosine
    fallback scores (rel_delta applies as before).
    """
    _check_embedder_meta(store, embedder)
    rows = store.indexed_chunks()
    if client:
        rows = [row for row in rows if row["client"] == client]
    if kind:
        rows = [row for row in rows if row["kind"] == kind]
    if not rows:
        return [], reranker is not None
    try:
        import numpy as np
    except ImportError as exc:
        raise EmbedderUnavailable(
            "numpy is not installed; install the optional dependency: pip install 'claude-tap[rag]'"
        ) from exc
    matrix = np.array([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
    embed_query = getattr(embedder, "embed_query", None) or embedder.embed
    query_vec = np.array(embed_query([query])[0], dtype=np.float32)
    scores = _cosine_scores(matrix, query_vec)
    cosine_by_id = {int(row["id"]): float(score) for row, score in zip(rows, scores)}
    by_id = {int(row["id"]): row for row in rows}
    # Vector channel: positive-overlap rows only, so zero-cosine tail rows
    # cannot ride the fusion into the candidate pool.
    vector_ranking = sorted(
        ((row_id, score) for row_id, score in cosine_by_id.items() if score > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )[:recall]
    candidate_ids = [
        cid
        for cid in _rrf_fuse([vector_ranking, *_fts_channels(store, "chunks", query, recall)], recall)
        if cid in by_id
    ]
    if not candidate_ids:
        return [], reranker is not None
    # Cross-snapshot dedup before reranking (no rerank compute wasted on dups);
    # cosine breaks ties exactly as it did for the pre-hybrid pipeline.
    deduped = _dedup_across_snapshots([(by_id[cid], cosine_by_id[cid]) for cid in candidate_ids])
    final, reranked = _final_scores(
        reranker,
        query,
        [row["text"] for row, _score, _bonus in deduped],
        [cosine_by_id[int(row["id"])] for row, _score, _bonus in deduped],
    )
    if reranked:
        kept = [
            (row, score, bonus)
            for (row, _cos, bonus), score in zip(deduped, final)
            if score > min_score
        ]
    else:
        kept = [
            (row, score, bonus)
            for (row, _cos, bonus), score in zip(deduped, final)
            if score >= min_score
        ]
        if kept:
            top = max(score for _, score, _ in kept)
            # Relative floor for uncalibrated cosine scores (rel_delta=1.0 disables).
            kept = [(row, score, bonus) for row, score, bonus in kept if score > top - rel_delta or score == top]
    groups: dict[int, SnapshotResult] = {}
    for row, score, bonus_sessions in kept:
        group = groups.setdefault(
            row["snapshot_id"],
            SnapshotResult(
                snapshot_id=row["snapshot_id"],
                client=row["client"],
                model=row["model"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                session_count=row["session_count"],
            ),
        )
        group.session_count += bonus_sessions
        group.hits.append(
            SearchHit(kind=row["kind"], title=row["title"] or "", text=row["text"], score=score)
        )
    ordered = sorted(
        groups.values(),
        key=lambda g: max((h.score for h in g.hits), default=0.0),
        reverse=True,
    )
    for group in ordered:
        group.hits = sorted(group.hits, key=lambda h: h.score, reverse=True)[:3]
    return ordered[:limit], reranked
```

`search_messages()` 整体替换为：

```python
def search_messages(
    store: KbStore,
    embedder: Embedder,
    query: str,
    *,
    client: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    rel_delta: float = 0.05,
    recall: int = 20,
    reranker: Reranker | None = None,
) -> tuple[list[SessionResult], bool]:
    """Hybrid search over indexed chat messages (user + assistant), grouped by session."""
    _check_embedder_meta(store, embedder)
    rows = store.indexed_messages()
    if client:
        rows = [row for row in rows if row["client"] == client]
    if not rows:
        return [], reranker is not None
    try:
        import numpy as np
    except ImportError as exc:
        raise EmbedderUnavailable(
            "numpy is not installed; install the optional dependency: pip install 'claude-tap[rag]'"
        ) from exc
    matrix = np.array([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
    embed_query = getattr(embedder, "embed_query", None) or embedder.embed
    query_vec = np.array(embed_query([query])[0], dtype=np.float32)
    scores = _cosine_scores(matrix, query_vec)
    cosine_by_id = {int(row["id"]): float(score) for row, score in zip(rows, scores)}
    by_id = {int(row["id"]): row for row in rows}
    vector_ranking = sorted(
        ((row_id, score) for row_id, score in cosine_by_id.items() if score > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )[:recall]
    candidate_ids = [
        cid
        for cid in _rrf_fuse([vector_ranking, *_fts_channels(store, "messages", query, recall)], recall)
        if cid in by_id
    ]
    if not candidate_ids:
        return [], reranker is not None
    candidates = [by_id[cid] for cid in candidate_ids]
    final, reranked = _final_scores(
        reranker,
        query,
        [row["text"] for row in candidates],
        [cosine_by_id[int(row["id"])] for row in candidates],
    )
    if reranked:
        kept = [(row, score) for row, score in zip(candidates, final) if score > min_score]
    else:
        kept = [(row, score) for row, score in zip(candidates, final) if score >= min_score]
        if kept:
            top = max(score for _, score in kept)
            kept = [(row, score) for row, score in kept if score > top - rel_delta or score == top]
    groups: dict[str, SessionResult] = {}
    for row, score in kept:
        group = groups.setdefault(
            row["session_id"],
            SessionResult(
                session_id=row["session_id"],
                client=row["client"],
                model=row["model"],
            ),
        )
        group.hits.append(
            MessageHit(text=row["text"], timestamp=row["timestamp"], score=float(score), role=row["role"])
        )
    ordered = sorted(
        groups.values(),
        key=lambda g: max((h.score for h in g.hits), default=0.0),
        reverse=True,
    )
    for group in ordered:
        group.hits = sorted(group.hits, key=lambda h: h.score, reverse=True)[:3]
    return ordered[:limit], reranked
```

`mcp_server.py` 适配：

1. import 区加 `from claude_tap.prompt_kb.rerank import Reranker, RerankerUnavailable, create_reranker`；`_ctx: tuple[KbStore, Embedder] | None` 改 `tuple[KbStore, Embedder, Reranker | None] | None`；`TYPE_CHECKING` 块加 `from claude_tap.prompt_kb.rerank import Reranker`（运行期 import 用于 create_reranker）。
2. `_get_ctx` 返回三元组，reranker 构建带降级：

```python
def _get_ctx() -> tuple[KbStore, Embedder, "Reranker | None"]:
    """Lazily open the KB store, build the embedder, and try the reranker.

    The embedding model loads on first tool call, not at server startup.
    FastMCP runs sync tools in a threadpool, so guard the one-time build
    with a lock (double-checked) to avoid loading the model twice.
    The reranker is best-effort: when it cannot load, search falls back to
    the fused ranking and reports reranked=false.
    """
    global _ctx
    if _ctx is None:
        with _ctx_lock:
            if _ctx is None:
                try:
                    reranker = create_reranker(load_config())
                except RerankerUnavailable:
                    reranker = None
                _ctx = (KbStore.default(), create_embedder(load_config()), reranker)
    return _ctx
```

3. `kb_search`：`store, embedder = _get_ctx()` 改 `store, embedder, reranker = _get_ctx()`；两个 search 调用加 `reranker=reranker` 并解包：

```python
        chunk_groups, chunks_reranked = search(
            store, embedder, query, client=client, kind=kind, limit=limit, min_score=min_score,
            rel_delta=rel_delta, reranker=reranker,
        )
        message_groups, messages_reranked = search_messages(
            store, embedder, query, client=client, limit=limit, min_score=min_score,
            rel_delta=rel_delta, reranker=reranker,
        )
```

4. 所有错误返回字典加 `"reranked": False`；成功返回字典加 `"reranked": chunks_reranked and messages_reranked`。
5. `kb_search` docstring 的 Returns 段改为：

```
    Returns:
        {"messages": [...], "chunks": [...], "reranked": bool} grouped by session / snapshot.
        The messages section is the primary evidence and comes first.
        Retrieval is hybrid (vector + keyword FTS channels, RRF-fused); when the local
        reranker model is unavailable, reranked=false and scores are fused-ranking fallbacks.
        rel_delta only applies to the fallback path.
        On embedder/reindex failure: {"error": str, "chunks": [], "messages": [], "reranked": false}.
```

`live.py` 适配：

1. import 区（`from claude_tap.prompt_kb.embed import ...` 行之后）加 `from claude_tap.prompt_kb.rerank import RerankerUnavailable, create_reranker`。
2. `_kb_embedder` 之后新增：

```python
    def _kb_reranker(self):
        """Process-wide lazy reranker; None when off or unavailable (degraded)."""
        if getattr(self, "_kb_reranker_instance", "unset") == "unset":
            try:
                self._kb_reranker_instance = create_reranker(load_config())
            except RerankerUnavailable:
                self._kb_reranker_instance = None
        return self._kb_reranker_instance
```

3. `_handle_kb_search`：min_score 解析块之后加 rel_delta 解析：

```python
        try:
            rel_delta = float(request.query.get("rel_delta", "") or 0.05)
        except ValueError:
            rel_delta = 0.05
```

两个调用改为：

```python
            results, chunks_reranked = kb_search(
                KbStore.default(),
                embedder,
                query,
                client=request.query.get("client") or None,
                kind=request.query.get("kind") or None,
                limit=int(request.query.get("limit", "10")),
                min_score=min_score,
                rel_delta=rel_delta,
                reranker=self._kb_reranker(),
            )
            messages, messages_reranked = kb_search_messages(
                KbStore.default(),
                embedder,
                query,
                client=request.query.get("client") or None,
                limit=int(request.query.get("limit", "10")),
                min_score=min_score,
                rel_delta=rel_delta,
                reranker=self._kb_reranker(),
            )
```

响应字典加 `"reranked": chunks_reranked and messages_reranked`（顶层，与 `"messages"` 并列）。

`cli.py` 适配：

1. import 区加 `from claude_tap.prompt_kb.rerank import RerankerUnavailable, create_reranker`。
2. `_embedder_or_exit` 之后加：

```python
def _reranker_or_none():
    try:
        return create_reranker(load_config())
    except RerankerUnavailable:
        return None
```

3. search 命令段：

```python
    reranker = _reranker_or_none()
    try:
        results, chunks_reranked = search(
            store, embedder, args.query, client=args.client, kind=args.kind, limit=args.limit,
            rel_delta=args.rel_delta, reranker=reranker,
        )
    except ReindexRequired as exc:
        print(str(exc), file=sys.stderr)
        return 3
    message_results, messages_reranked = search_messages(
        store, embedder, args.query, client=args.client, limit=args.limit,
        rel_delta=args.rel_delta, reranker=reranker,
    )
    print(f"reranked: {'yes' if chunks_reranked and messages_reranked else 'no'}")
```

（原 `try: results = search(...)` 结构保持，只是解包与传参变化；`reranked:` 行在 `messages:` 段之前打印。）

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/prompt_kb/ -q`
Expected: all passed（既有用例机械适配后语义不变，新增管线用例通过）

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/search.py claude_tap/prompt_kb/mcp_server.py claude_tap/live.py claude_tap/prompt_kb/cli.py tests/prompt_kb/
git commit -m "feat(prompt_kb): 混合检索管线——三通道 RRF 融合 + reranker 校准重排，全端 reranked 透传

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: 状态面——reranker_status 接入 kb_status / CLI status / docstring 收尾

**Files:**
- Modify: `claude_tap/prompt_kb/mcp_server.py`（kb_status）
- Modify: `claude_tap/prompt_kb/cli.py`（status 命令）
- Test: `tests/prompt_kb/test_mcp_server.py`、`tests/prompt_kb/test_cli.py`

**Interfaces:**
- Consumes: `reranker_status(config)`（Task 4）

- [ ] **Step 1: Write the failing tests**

`test_mcp_server.py` 追加：

```python
def test_kb_status_includes_reranker_state(ctx):
    result = mcp_server.kb_status()
    # sentence-transformers is not installed in the test venv: configured on -> unavailable
    assert result["reranker"] in ("unavailable", "BAAI/bge-reranker-base", "off")
```

`test_cli.py` 追加：

```python
def test_kb_status_prints_reranker_line(seeded_kb, capsys):
    assert kb_main(["status"]) == 0
    assert "reranker=" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/prompt_kb/test_mcp_server.py -k reranker_state tests/prompt_kb/test_cli.py -k reranker_line -v`
Expected: FAIL（KeyError / 无 reranker= 行）

- [ ] **Step 3: Write the implementation**

`mcp_server.py`：import 区加 `from claude_tap.prompt_kb.rerank import reranker_status`；`kb_status()` 返回改为：

```python
        store = KbStore.default()
        return {
            **store.stats(),
            "embedder": store.get_meta("embedder_name") or "none",
            "reranker": reranker_status(load_config()),
        }
```

`kb_status` docstring Returns 段改为：

```
    Returns:
        store.stats() keys (snapshots/chunks/pending/failed/indexed/messages/messages_user/
        messages_assistant) plus "embedder" (indexed embedder name, or "none") and
        "reranker" ("off" | "unavailable" | reranker model name).
```

`cli.py`：import 加 `from claude_tap.prompt_kb.rerank import reranker_status`；status 段在 `print(f"embedder=...")` 后加：

```python
        print(f"reranker={reranker_status(load_config())}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/prompt_kb/test_mcp_server.py tests/prompt_kb/test_cli.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/mcp_server.py claude_tap/prompt_kb/cli.py tests/prompt_kb/test_mcp_server.py tests/prompt_kb/test_cli.py
git commit -m "feat(prompt_kb): kb_status 与 CLI status 输出 reranker 状态

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: CLI rebuild-fts 子命令

**Files:**
- Modify: `claude_tap/prompt_kb/cli.py`
- Test: `tests/prompt_kb/test_cli.py`

**Interfaces:**
- Consumes: `KbStore.rebuild_fts()`（Task 3）

- [ ] **Step 1: Write the failing test**

`test_cli.py` 追加：

```python
def test_kb_rebuild_fts(seeded_kb, capsys):
    assert kb_main(["rebuild-fts"]) == 0
    assert "fts_rebuilt=1" in capsys.readouterr().out
    # Idempotent: second run rebuilds the same single chunk row.
    assert kb_main(["rebuild-fts"]) == 0
    assert "fts_rebuilt=1" in capsys.readouterr().out
    # Keyword search still resolves after the rebuild.
    assert seeded_kb.fts_rank("chunks", "tri", "sandbox", 10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/prompt_kb/test_cli.py -k rebuild_fts -v`
Expected: FAIL（argparse: invalid choice: 'rebuild-fts'）

- [ ] **Step 3: Write the implementation**

`_build_parser()` 中 `sub.add_parser("reindex")` 行后加：

```python
    sub.add_parser("rebuild-fts")
```

`kb_main` 中 status 分支之后加（在 embedder 构建之前，rebuild 不需要模型）：

```python
    if args.command == "rebuild-fts":
        print(f"fts_rebuilt={store.rebuild_fts()}")
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/prompt_kb/test_cli.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/prompt_kb/cli.py tests/prompt_kb/test_cli.py
git commit -m "feat(prompt_kb): CLI 新增 kb rebuild-fts——jieba 后装升级与索引自救

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: dashboard 降级提示 + i18n + README

**Files:**
- Modify: `claude_tap/dashboard.html`（kbSearch、renderKbResults、kbRenderFiltered、i18n 两处、placeholder、CSS）
- Modify: `README.md:636`（rag 段）
- Test: `tests/prompt_kb/test_kb_render_logic.py`（追加锚点与逻辑用例）

**Interfaces:**
- Consumes: HTTP 响应的 `reranked` 字段（Task 5）

- [ ] **Step 1: Write the failing tests**

`test_kb_render_logic.py` 追加：

```python
def test_template_contains_rerank_notice_and_i18n():
    html = read_dashboard_template()
    assert "kbLastReranked" in html
    assert "kb_rerank_degraded" in html
    assert "kb-rerank-notice" in html
    # both locales
    assert '"Reranker unavailable' in html or "Reranker unavailable" in html
    assert "重排不可用" in html


def test_render_kb_results_accepts_reranked_param():
    html = read_dashboard_template()
    assert "function renderKbResults(results, messages = [], reranked = true)" in html
    assert "renderKbResults(data.results || [], data.messages || [], data.reranked !== false)" in html


def test_search_placeholder_mentions_hybrid():
    html = read_dashboard_template()
    assert "Hybrid keyword + semantic search" in html
    assert "关键词与语义混合搜索" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/prompt_kb/test_kb_render_logic.py -v`
Expected: 3 个新用例 FAIL

- [ ] **Step 3: Write the implementation**

`dashboard.html` 改动：

1. JS 状态区（`let kbLastMessages = null;` 行后）加：

```javascript
let kbLastReranked = true;
```

2. `renderKbResults` 改为：

```javascript
function renderKbResults(results, messages = [], reranked = true) {
  kbLastGroups = results || [];
  kbLastMessages = messages || [];
  kbLastReranked = reranked;
  kbRenderFiltered();
}
```

3. `kbSearch` 中错误分支（`kbLastMessages = [];` 之后）加 `kbLastReranked = true;`；成功行改为：

```javascript
  renderKbResults(data.results || [], data.messages || [], data.reranked !== false);
```

4. `kbRenderFiltered` 中，两个 empty-state 早退之后、`if (msgGroups.length)` 之前加：

```javascript
  if (!kbLastReranked) {
    const notice = document.createElement("div");
    notice.className = "kb-rerank-notice";
    notice.textContent = t("kb_rerank_degraded");
    container.appendChild(notice);
  }
```

5. CSS（`.kb-summary` 规则附近）加：

```css
.kb-rerank-notice { font-size: 12px; color: #8a6d1d; background: #fdf3d7; border: 1px solid #f0e0a8; border-radius: var(--radius-sm); padding: 6px 10px; margin-bottom: 8px; }
```

6. i18n en 区（`kb_role_user` 行附近）加：

```javascript
    kb_rerank_degraded: "Reranker unavailable — fused keyword + semantic ranking.",
```

   zh 区对应位置加：

```javascript
    kb_rerank_degraded: "重排不可用——按关键词+语义融合分数排序。",
```

7. placeholder：en `kb_search_placeholder: "Search prompt rules or tool definitions…"` 改 `"Hybrid keyword + semantic search…"`；zh 改 `"关键词与语义混合搜索…"`；`#kb-query` 的静态 `placeholder=` 属性同步改为 `"关键词与语义混合搜索…"`。

`README.md` 636 行段落末尾（`claude-tap kb status` 句后）追加一句：

```markdown
Search is hybrid (keyword FTS + vector, RRF-fused) with a local cross-encoder reranker (`BAAI/bge-reranker-base`, ~280MB one-time download; results carry `reranked: false` when it cannot load). Chinese keyword matching uses jieba segmentation — after installing jieba into an existing setup, run `claude-tap kb rebuild-fts` once to resegment the keyword index.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/prompt_kb/test_kb_render_logic.py tests/prompt_kb/test_kb_page.py tests/prompt_kb/test_kb_render_browser.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add claude_tap/dashboard.html README.md tests/prompt_kb/test_kb_render_logic.py
git commit -m "feat(prompt_kb): dashboard 重排降级提示与混合搜索文案，README 补 reranker 说明

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: 全量回归 + 真实库验证 + spec 实施验证附录

**Files:**
- Modify: `docs/superpowers/specs/2026-08-13-hybrid-search-reranker-design.md`（追加 `## 实施验证` 段）

**Interfaces:**
- Consumes: 全部前序任务

- [ ] **Step 1: 全量回归**

Run: `.venv/bin/pytest tests/ --ignore=tests/test_e2e.py -q`
Expected: 全绿（既有 1158+ 加上新增用例）
Run: `.venv/bin/pytest tests/test_e2e.py -q --timeout=120`
Expected: 68 passed

- [ ] **Step 2: uv tool 环境补 jieba 并重装**

真实库验证走 uv tool editable 安装（`.venv` 无 `[rag]`，沿用上个计划的先例）：

```bash
cd /usr/local/data/apps/claude-tap-main
uv tool install --force --editable '.[mcp,rag]'
claude-tap kb status   # 应见 reranker=BAAI/bge-reranker-base 与 messages_user/messages_assistant 计数
```

- [ ] **Step 3: 真实库迁移与 FTS 回填验证**

```bash
cp ~/.local/share/claude-tap/prompt_kb.sqlite3 /tmp/prompt_kb_backup_pre_hybrid.sqlite3
claude-tap kb rebuild-fts   # 输出 fts_rebuilt=<chunks+messages 总数>
claude-tap kb status        # 再次确认计数与 reranker 状态
```

- [ ] **Step 4: 7 组参考查询验收**

逐条跑 `claude-tap kb search "<query>"`（首次触发 reranker 模型下载 ~280MB；若 HF 直连失败，用 `HF_ENDPOINT=https://hf-mirror.com claude-tap kb search ...` 重试）：

1. `哪个 CLI 有沙箱 shell 工具`
2. `which CLI has a sandboxed shell tool`
3. `怎么写 commit message`
4. `前端页面截图验证`
5. `取消定时任务 cron`（kind 过滤用 `--kind tool` 变体也跑一次）
6. `沙箱 sandbox 执行命令`
7. `Playwright 浏览器截图验证`

验收标准（写进附录表格）：

- 每次输出头部 `reranked: yes`
- Q1：Bash 相关命中进入 chunks 前列（对比基线 getDiagnostics 压 Bash 的遗留问题，记录是否修正）
- Q5：CronDelete 仍居首且为字面命中
- Q6/Q2：无样板 section 霸榜（回归保护）
- 至少一组中文查询的 messages 区出现关键词独有召回（向量此前排不上的字面命中）

- [ ] **Step 5: 追加 spec 实施验证并提交**

在 spec 末尾追加 `## 实施验证` 段：日期、回归结果、真实库 fts_rebuilt 计数、reranker 状态、7 组查询验收表、遗留问题（如有）。然后：

```bash
git add docs/superpowers/specs/2026-08-13-hybrid-search-reranker-design.md
git commit -m "docs(prompt_kb): 混合检索与重排实施验证记录

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review 记录

- **Spec 覆盖**：§1→Task 2/3，§2→Task 1，§3→Task 5，§4→Task 4/5，§5→Task 5/6/7/8，§6→Task 2/5 降级矩阵，§7→各任务测试 + Task 9 验收。全覆盖。
- **类型一致性**：`fts_rank(entity, tokenizer, match_query, limit)` 在 Task 2/3/5 一致；`reranked` 字段在 Task 5/6/8 一致；`_final_scores` 返回 `(scores, reranked)` 与两函数用法一致。
- **已知取舍**：test_search_messages.py 未逐行给出适配代码（文件未逐行读取），实现者按 test_search.py 的解包模式机械适配——已在任务内明确标注。
