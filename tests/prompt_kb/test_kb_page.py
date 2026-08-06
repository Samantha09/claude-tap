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
