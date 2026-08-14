import pytest

numpy = pytest.importorskip("numpy")  # search depends on the [rag] extra

from claude_tap.prompt_kb.index import ensure_embedder_meta, index_pending  # noqa: E402
from claude_tap.prompt_kb.search import ReindexRequired, _match_query, _rrf_fuse, search  # noqa: E402
from claude_tap.prompt_kb.store import KbStore  # noqa: E402
from tests.prompt_kb.fake_embedder import FakeEmbedder  # noqa: E402
from tests.prompt_kb.fake_reranker import FakeReranker  # noqa: E402


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
    results, _ = search(store, FakeEmbedder(), "shell sandbox")
    assert results[0].client == "codex"
    assert results[0].hits[0].title == "shell"
    assert results[0].hits[0].score > 0.5


def test_search_filters_by_client_and_kind(trace_db):
    store = _indexed_store()
    # "sandbox" only appears in the codex chunk; filtering to claude-code
    # leaves no relevant hit. (FakeEmbedder buckets: sandbox=7, and the
    # claude-code chunk tokens hash to buckets 0/4/14 — no overlap.)
    assert search(store, FakeEmbedder(), "sandbox", client="claude-code")[0] == []
    results, _ = search(store, FakeEmbedder(), "prose", kind="prompt_section")
    assert all(h.kind == "prompt_section" for r in results for h in r.hits)


def test_search_empty_index_returns_empty(trace_db):
    assert search(KbStore.default(), FakeEmbedder(), "anything")[0] == []


def test_search_detects_embedder_mismatch(trace_db):
    store = _indexed_store()

    class OtherEmbedder(FakeEmbedder):
        name = "other"

    with pytest.raises(ReindexRequired):
        search(store, OtherEmbedder(), "shell")


def test_search_respects_min_score(trace_db):
    store = _indexed_store()
    all_hits, _ = search(store, FakeEmbedder(), "shell sandbox", min_score=0.0)
    total = sum(len(g.hits) for g in all_hits)
    filtered, _ = search(store, FakeEmbedder(), "shell sandbox", min_score=0.999)
    kept = sum(len(g.hits) for g in filtered)
    assert total > 0
    assert kept < total
    assert all(h.score >= 0.999 for g in filtered for h in g.hits)


def _seed_overlap(store: KbStore) -> None:
    """三个快照：A 与 query 全量重叠（top），B/C 部分重叠（长尾）。"""
    for client, model, seen, chunks in [
        (
            "codex",
            "gpt-5",
            "2026-08-01T00:00:00Z",
            [("prompt_section", "Guide", "alpha beta gamma delta epsilon zeta runs fast")],
        ),
        (
            "claude-code",
            "claude",
            "2026-08-02T00:00:00Z",
            [("prompt_section", "Notes", "alpha only shares one token here")],
        ),
        ("claude-code", "claude", "2026-08-03T00:00:00Z", [("prompt_section", "Style", "write elegant prose")]),
    ]:
        sid, _ = store.upsert_snapshot(
            content_hash=f"h-{client}-{seen}",
            client=client,
            provider="p",
            model=model,
            system_prompt="s",
            developer_prompt="",
            tools_json="[]",
            seen_at=seen,
        )
        store.replace_chunks(sid, chunks)


def test_rel_delta_trims_long_tail(trace_db):
    store = KbStore.default()
    _seed_overlap(store)
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    results, _ = search(store, embedder, "alpha beta gamma delta epsilon zeta")
    assert len(results) == 1  # top=0.938，0.667/0.510 被 rel_delta=0.05 截断
    assert results[0].client == "codex"
    everything, _ = search(store, embedder, "alpha beta gamma delta epsilon zeta", rel_delta=1.0)
    assert len(everything) == 3


def test_identical_chunks_folded_across_snapshots(trace_db):
    store = KbStore.default()
    for client, seen in [("codex", "2026-08-01T00:00:00Z"), ("claude-code", "2026-08-02T00:00:00Z")]:
        sid, _ = store.upsert_snapshot(
            content_hash=f"h-{client}",
            client=client,
            provider="p",
            model="m",
            system_prompt="s",
            developer_prompt="",
            tools_json="[]",
            seen_at=seen,
        )
        store.replace_chunks(sid, [("tool", "shell", "sandbox shell command runner")])
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    results, _ = search(store, embedder, "shell sandbox", rel_delta=1.0)
    assert len(results) == 1  # 同分平局归最新 last_seen 的快照
    assert results[0].client == "claude-code"
    assert results[0].session_count == 2  # 被折叠快照的计数累加


def test_match_query_sanitizes():
    assert _match_query('怎么取消 "cron" (定时任务)?') == "怎么取消 OR cron OR 定时任务"
    assert _match_query("!!!") == ""


def test_rrf_fusion_prefers_multi_channel_hits():
    fused = _rrf_fuse([[(1, 0.9), (2, 0.8)], [(2, 5.0), (3, 4.0)]], 10)
    assert fused[0] == 2  # present in both channels
    assert set(fused) == {1, 2, 3}


def _seed_chinese(store: KbStore) -> None:
    sid, _ = store.upsert_snapshot(
        content_hash="h-zh",
        client="codex",
        provider="p",
        model="m",
        system_prompt="s",
        developer_prompt="",
        tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
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
        content_hash="h-cron",
        client="codex",
        provider="p",
        model="m",
        system_prompt="s",
        developer_prompt="",
        tools_json="[]",
        seen_at="2026-08-01T00:00:00Z",
    )
    store.replace_chunks(sid, [("tool", "cron", "CronDelete cancels scheduled cron jobs")])
    embedder = FakeEmbedder()
    ensure_embedder_meta(store, embedder)
    index_pending(store, embedder)
    # "CronDele" is not a whole token: zero FakeEmbedder overlap, trigram substring hit.
    results, _ = search(store, embedder, "CronDele")
    assert len(results) == 1
    assert results[0].hits[0].title == "cron"


def test_reranked_neutral_hits_dropped_by_default(trace_db):
    store = _indexed_store()
    # FakeReranker: each chunk overlaps exactly half the query tokens → 0.5,
    # the sigmoid neutral zone of a calibrated reranker (boilerplate band).
    results, reranked = search(store, FakeEmbedder(), "shell prose", reranker=FakeReranker())
    assert reranked is True
    assert [h.score for g in results for h in g.hits] == []
    # Explicit min_score=0.0 opts out of the neutral floor.
    kept, _ = search(store, FakeEmbedder(), "shell prose", min_score=0.0, reranker=FakeReranker())
    assert sorted(h.score for g in kept for h in g.hits) == [pytest.approx(0.5), pytest.approx(0.5)]


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
    results, reranked = search(
        store, embedder, "alpha beta gamma delta epsilon zeta", min_score=0.0, reranker=FakeReranker()
    )
    assert reranked is True
    # FakeReranker scores: A=1.0 (full overlap), B=1/6 ("alpha"), C=0.0 (dropped
    # by the strict reranked filter). B survives despite being far below the
    # top score: rel_delta is not applied to calibrated reranker scores.
    # min_score=0.0 opts out of the default neutral floor so B is visible.
    assert len(results) == 2
    scores = sorted(h.score for r in results for h in r.hits)
    assert scores == [pytest.approx(1 / 6), pytest.approx(1.0)]
