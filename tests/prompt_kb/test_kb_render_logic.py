"""Python replica of the dashboard KB fold/filter pipeline.

Mirrors kbBestScore / kbFilterHits / kbFoldGroups / kbScoreClass in
claude_tap/dashboard.html (js-in-html-testing Layer 1).
"""

from claude_tap.dashboard import read_dashboard_template


def best_score(group):
    return max((h["score"] for h in group.get("hits", [])), default=0.0)


def filter_hits(group, min_score):
    return {**group, "hits": [h for h in group.get("hits", []) if h["score"] >= min_score]}


def fold_groups(groups):
    by_key = {}
    for g in groups:
        by_key.setdefault((g["client"], g["model"]), []).append(g)
    folded = []
    for lst in by_key.values():
        ordered = sorted(lst, key=lambda g: g["first_seen"], reverse=True)
        folded.append({"main": ordered[0], "history": ordered[1:]})
    folded.sort(key=lambda f: best_score(f["main"]), reverse=True)
    return folded


def score_class(score):
    if score >= 0.9:
        return "score-high"
    if score >= 0.7:
        return "score-mid"
    return "score-low"


def _group(client, model, first_seen, scores):
    return {
        "client": client,
        "model": model,
        "first_seen": first_seen,
        "hits": [{"kind": "tool", "title": "T", "text": "x", "score": s} for s in scores],
    }


def test_js_functions_exist_in_template():
    html = read_dashboard_template()
    for fn in ("kbBestScore", "kbFilterHits", "kbFoldGroups", "kbScoreClass"):
        assert f"function {fn}(" in html, fn


def test_fold_merges_same_client_model_newest_first():
    groups = [
        _group("claude", "k3", "2026-08-05T10:00:00", [0.8]),
        _group("claude", "k3", "2026-08-06T16:00:00", [0.9]),
        _group("codex", "gpt5", "2026-08-04T09:00:00", [0.7]),
    ]
    folded = fold_groups(groups)
    assert len(folded) == 2
    claude = next(f for f in folded if f["main"]["client"] == "claude")
    assert claude["main"]["first_seen"] == "2026-08-06T16:00:00"
    assert [g["first_seen"] for g in claude["history"]] == ["2026-08-05T10:00:00"]
    codex = next(f for f in folded if f["main"]["client"] == "codex")
    assert codex["history"] == []


def test_fold_orders_by_main_best_score():
    groups = [
        _group("claude", "k3", "2026-08-06T16:00:00", [0.75]),
        _group("codex", "gpt5", "2026-08-04T09:00:00", [0.95]),
    ]
    folded = fold_groups(groups)
    assert folded[0]["main"]["client"] == "codex"


def test_filter_hits_drops_below_threshold():
    g = _group("claude", "k3", "2026-08-06T16:00:00", [0.82, 0.91, 0.65])
    assert [h["score"] for h in filter_hits(g, 0.7)["hits"]] == [0.82, 0.91]
    assert filter_hits(g, 0.95)["hits"] == []
    assert [h["score"] for h in filter_hits(g, 0.0)["hits"]] == [0.82, 0.91, 0.65]


def filter_message_groups(groups, min_score):
    return [
        {**g, "hits": [h for h in g.get("hits", []) if h["score"] >= min_score]}
        for g in groups
        if any(h["score"] >= min_score for h in g.get("hits", []))
    ]


def test_score_class_thresholds():
    assert score_class(0.95) == "score-high"
    assert score_class(0.9) == "score-high"
    assert score_class(0.82) == "score-mid"
    assert score_class(0.7) == "score-mid"
    assert score_class(0.69) == "score-low"


def test_filter_message_groups():
    groups = [
        {"session_id": "s1", "hits": [{"score": 0.9}, {"score": 0.3}]},
        {"session_id": "s2", "hits": [{"score": 0.2}]},
    ]
    filtered = filter_message_groups(groups, 0.5)
    assert [g["session_id"] for g in filtered] == ["s1"]
    assert len(filtered[0]["hits"]) == 1


def test_template_contains_message_rendering():
    from claude_tap.dashboard import read_dashboard_template

    html = read_dashboard_template()
    assert "kbRenderMessageCard" in html
    assert "kb_messages_section" in html
    assert "kb_view_session" in html
    assert "kbFilterMessageGroups" in html
