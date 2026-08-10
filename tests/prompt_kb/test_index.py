import array

from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending, rebuild_index
from claude_tap.prompt_kb.store import KbStore
from tests.prompt_kb.fake_embedder import FakeEmbedder


def _seed(store: KbStore) -> None:
    snap_id, _ = store.upsert_snapshot(
        content_hash="h",
        client="codex",
        provider="openai",
        model="gpt-5",
        system_prompt="s",
        developer_prompt="",
        tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(snap_id, [("tool", "a", "alpha text"), ("tool", "b", "beta text")])


def test_index_pending_embeds_and_marks(trace_db):
    store = KbStore.default()
    _seed(store)
    result = index_pending(store, FakeEmbedder())
    assert result == {"indexed": 2, "failed": 0, "messages_indexed": 0, "messages_failed": 0, "remaining": 0}
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
    assert result == {"indexed": 0, "failed": 2, "messages_indexed": 0, "messages_failed": 0, "remaining": 0}
    assert store.stats()["failed"] == 2


class _FailOnceEmbedder(FakeEmbedder):
    """Fails on the first embed call, then behaves normally."""

    def __init__(self):
        super().__init__()
        self._failed = False

    def embed(self, texts):
        if not self._failed:
            self._failed = True
            raise RuntimeError("transient boom")
        return super().embed(texts)


def test_failed_chunk_retried_next_round(trace_db):
    store = KbStore.default()
    _seed(store)
    embedder = _FailOnceEmbedder()
    first = index_pending(store, embedder, batch_size=1)
    assert first == {"indexed": 1, "failed": 1, "messages_indexed": 0, "messages_failed": 0, "remaining": 0}
    assert store.stats()["failed"] == 1
    second = index_pending(store, embedder, batch_size=1)
    assert second == {"indexed": 1, "failed": 0, "messages_indexed": 0, "messages_failed": 0, "remaining": 0}
    assert store.stats()["failed"] == 0
    assert store.stats()["indexed"] == 2


def test_chunk_failing_three_times_stays_failed(trace_db):
    store = KbStore.default()
    _seed(store)
    for _ in range(3):
        index_pending(store, _FlakyEmbedder(), batch_size=2)
    stats = store.stats()
    assert stats["failed"] == 2
    # Beyond the retry cap the chunks are no longer requeued.
    result = index_pending(store, FakeEmbedder(), batch_size=2)
    assert result["indexed"] == 0
    assert store.stats()["failed"] == 2


def test_rebuild_resets_then_reindexes(trace_db):
    store = KbStore.default()
    _seed(store)
    index_pending(store, FakeEmbedder())
    assert rebuild_index(store, FakeEmbedder())["indexed"] == 2
    assert store.stats()["indexed"] == 2


def test_index_pending_embeds_messages(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(
        session_id="s1", record_index=0, message_index=0,
        client="claude", model="k3", timestamp="t",
        content_hash="h1", text="how to fix race condition",
        seen_at="t",
    )
    embedder = FakeEmbedder()
    result = index_pending(store, embedder)
    assert result["messages_indexed"] == 1
    assert result["messages_failed"] == 0
    assert len(store.indexed_messages()) == 1


def test_index_pending_message_batch_failure_marks_failed(tmp_path):
    class FailingEmbedder:
        name = "fail"
        dimension = 16

        def embed(self, texts):
            raise RuntimeError("boom")

    store = KbStore(tmp_path / "kb.sqlite3")
    store.upsert_message(
        session_id="s1", record_index=0, message_index=0,
        client="claude", model="k3", timestamp="t",
        content_hash="h1", text="x", seen_at="t",
    )
    result = index_pending(store, FailingEmbedder())
    assert result["messages_failed"] == 1


def test_rebuild_resets_both_tables(tmp_path):
    store = KbStore(tmp_path / "kb.sqlite3")
    snap_id, _ = store.upsert_snapshot(
        content_hash="h", client="c", provider="p", model="m",
        system_prompt="s", developer_prompt="", tools_json="[]", seen_at="t",
    )
    store.replace_chunks(snap_id, [("tool", "t", "text")])
    store.upsert_message(
        session_id="s1", record_index=0, message_index=0,
        client="c", model="m", timestamp="t",
        content_hash="h1", text="msg", seen_at="t",
    )
    embedder = FakeEmbedder()
    index_pending(store, embedder)
    result = rebuild_index(store, embedder)
    assert result["indexed"] == 1 and result["messages_indexed"] == 1


def test_ensure_embedder_meta(trace_db):
    store = KbStore.default()
    ensure_embedder_meta(store, FakeEmbedder())
    assert store.get_meta("embedder_name") == "fake"
    assert store.get_meta("embedding_dim") == "16"


class _LateBindingEmbedder(FakeEmbedder):
    """Dimension unknown until the first successful embed, like ApiEmbedder."""

    def __init__(self):
        self.dimension = 0

    def embed(self, texts):
        self.dimension = 16
        return super().embed(texts)


def test_index_pending_records_late_bound_dimension(trace_db):
    store = KbStore.default()
    _seed(store)
    result = index_pending(store, _LateBindingEmbedder())
    assert result["indexed"] == 2
    assert store.get_meta("embedding_dim") == "16"
