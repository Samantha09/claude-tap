import pytest

numpy = pytest.importorskip("numpy")  # search depends on the [rag] extra

from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending  # noqa: E402
from claude_tap.prompt_kb.search import ReindexRequired, search  # noqa: E402
from claude_tap.prompt_kb.store import KbStore  # noqa: E402
from tests.prompt_kb.fake_embedder import FakeEmbedder  # noqa: E402


def _seed(store: KbStore) -> None:
    a, _ = store.upsert_snapshot(
        content_hash="ha",
        client="codex",
        provider="openai",
        model="gpt-5",
        system_prompt="s",
        developer_prompt="",
        tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(a, [("tool", "shell", "sandbox shell command runner")])
    b, _ = store.upsert_snapshot(
        content_hash="hb",
        client="claude-code",
        provider="anthropic",
        model="claude",
        system_prompt="s",
        developer_prompt="",
        tools_json="[]",
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


def test_search_respects_min_score(trace_db):
    store = _indexed_store()
    all_hits = search(store, FakeEmbedder(), "shell sandbox", min_score=0.0)
    total = sum(len(g.hits) for g in all_hits)
    filtered = search(store, FakeEmbedder(), "shell sandbox", min_score=0.999)
    kept = sum(len(g.hits) for g in filtered)
    assert total > 0
    assert kept < total
    assert all(h.score >= 0.999 for g in filtered for h in g.hits)


def _seed_overlap(store: KbStore) -> None:
    """三个快照：A 与 query 全量重叠（top），B/C 部分重叠（长尾）。"""
    for client, model, seen, chunks in [
        ("codex", "gpt-5", "2026-08-01T00:00:00Z",
         [("prompt_section", "Guide", "alpha beta gamma delta epsilon zeta runs fast")]),
        ("claude-code", "claude", "2026-08-02T00:00:00Z",
         [("prompt_section", "Notes", "alpha only shares one token here")]),
        ("claude-code", "claude", "2026-08-03T00:00:00Z",
         [("prompt_section", "Style", "write elegant prose")]),
    ]:
        sid, _ = store.upsert_snapshot(
            content_hash=f"h-{client}-{seen}", client=client, provider="p", model=model,
            system_prompt="s", developer_prompt="", tools_json="[]", seen_at=seen,
        )
        store.replace_chunks(sid, chunks)


def test_rel_delta_trims_long_tail(trace_db):
    store = KbStore.default()
    _seed_overlap(store)
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    results = search(store, embedder, "alpha beta gamma delta epsilon zeta")
    assert len(results) == 1  # top=0.938，0.667/0.510 被 rel_delta=0.05 截断
    assert results[0].client == "codex"
    everything = search(store, embedder, "alpha beta gamma delta epsilon zeta", rel_delta=1.0)
    assert len(everything) == 3


def test_identical_chunks_folded_across_snapshots(trace_db):
    store = KbStore.default()
    for client, seen in [("codex", "2026-08-01T00:00:00Z"), ("claude-code", "2026-08-02T00:00:00Z")]:
        sid, _ = store.upsert_snapshot(
            content_hash=f"h-{client}", client=client, provider="p", model="m",
            system_prompt="s", developer_prompt="", tools_json="[]", seen_at=seen,
        )
        store.replace_chunks(sid, [("tool", "shell", "sandbox shell command runner")])
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    results = search(store, embedder, "shell sandbox", rel_delta=1.0)
    assert len(results) == 1  # 同分平局归最新 last_seen 的快照
    assert results[0].client == "claude-code"
    assert results[0].session_count == 2  # 被折叠快照的计数累加
