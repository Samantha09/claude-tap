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
    assert "搜索 prompt 规则或工具定义" in html
    assert '"kb_view"' in html or "kb_view" in html
    # zh-CN entries exist alongside en fallbacks
    assert "Prompt 知识库" in html


def test_refresh_sessions_preserves_kb_view():
    html = read_dashboard_template()
    # SSE/polling refreshes must not bounce the user off the KB tab.
    assert 'state.view !== "detail" && state.view !== "kb" && state.view !== "stats"' in html
    assert 'state.view !== "kb" && state.view !== "stats" && (!preserveSelection || !stillVisible)' in html
    assert '} else if (state.view !== "kb" && state.view !== "stats") {' in html


def test_kb_timeline_toggles_instead_of_stacking():
    html = read_dashboard_template()
    assert 'card.querySelector("ul.kb-timeline")' in html
    assert 'list.className = "kb-timeline"' in html
    assert "fmtTime(v.first_seen)" in html


def test_kb_status_uses_short_embedder_label():
    html = read_dashboard_template()
    assert "kbEmbedderLabel(data.embedder)" in html
    assert "fmtTime(group.first_seen)" in html


def test_kb_new_i18n_entries_bilingual():
    html = read_dashboard_template()
    for key in (
        "kb_min_score",
        "kb_expand",
        "kb_collapse",
        "kb_latest",
        "kb_history_snapshots",
        "kb_summary",
        "kb_summary_folded",
        "kb_no_results_title",
        "kb_no_results_hint",
        "kb_filtered_by_score",
    ):
        assert f"{key}:" in html, key
    # zh-CN
    for text in ("最低分数", "展开全文", "收起", "最新版", "个历史快照", "找到 {n} 组结果", "知识库只收录"):
        assert text in html, text
    # en
    for text in ("Min score", "Show full text", "Collapse", "Latest", "history snapshot", "result group"):
        assert text in html, text


def test_kb_controls_have_min_score_slider():
    html = read_dashboard_template()
    assert 'id="kb-min-score"' in html
    assert 'type="range"' in html
    assert 'id="kb-min-score-val"' in html


def test_kb_new_css_classes_present():
    html = read_dashboard_template()
    for cls in (
        ".kb-status-chips",
        ".kb-chip",
        ".kb-summary",
        ".kb-empty",
        ".kb-group-header",
        ".kb-latest-badge",
        ".kb-group-meta",
        ".kb-hit-head",
        ".kb-hit-kind.tool",
        ".kb-hit-kind.prompt",
        ".kb-score-bar",
        ".kb-score-fill",
        ".score-high",
        ".score-mid",
        ".score-low",
        ".kb-hit-text",
        ".kb-expand-btn",
        ".kb-history-fold",
        ".kb-timeline-current",
        ".kb-btn-primary",
        ".kb-btn-secondary",
    ):
        assert cls in html, cls


def test_kb_render_pipeline_wired():
    html = read_dashboard_template()
    for fn in ("kbRenderFiltered", "kbRenderGroupCard", "kbRenderHit", "kbEmptyState"):
        assert f"function {fn}(" in html, fn
    assert "kbLastGroups" in html
    assert '$("#kb-min-score").addEventListener("input"' in html
    assert "kbFoldGroups(filtered)" in html


def test_kb_timeline_current_uses_class():
    html = read_dashboard_template()
    assert 'item.classList.add("kb-timeline-current")' in html
