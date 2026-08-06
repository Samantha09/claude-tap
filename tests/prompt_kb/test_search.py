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
    # "sandbox" only appears in the codex chunk; filtering to claude-code
    # leaves no relevant hit. (FakeEmbedder buckets: sandbox=7, and the
    # claude-code chunk tokens hash to buckets 0/4/14 — no overlap.)
    assert search(store, FakeEmbedder(), "sandbox", client="claude-code") == []
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
