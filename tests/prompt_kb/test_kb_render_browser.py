"""Browser tests for the dashboard KB result rendering (js-in-html-testing Layer 2).

Loads the real dashboard template via file://, injects mock search results,
and drives renderKbResults / kbRenderFiltered directly. Top-level dashboard
init fetches fail on file:// but all function declarations are hoisted and
callable.
"""

import json

import pytest

from claude_tap.dashboard import read_dashboard_template

pytest.importorskip("playwright")

LONG_TEXT = "Read tool description. " * 30  # > 200 chars

GROUPS = [
    {
        "snapshot_id": 1, "client": "claude", "model": "k3-256k",
        "first_seen": "2026-08-05T10:28:53", "last_seen": "2026-08-05T10:28:53",
        "session_count": 1,
        "hits": [{"kind": "tool", "title": "Read", "text": LONG_TEXT, "score": 0.823}],
    },
    {
        "snapshot_id": 2, "client": "claude", "model": "k3-256k",
        "first_seen": "2026-08-06T16:34:47", "last_seen": "2026-08-06T16:34:47",
        "session_count": 1,
        "hits": [
            {"kind": "tool", "title": "Read", "text": LONG_TEXT, "score": 0.95},
            {"kind": "prompt_section", "title": "Rules", "text": "short rule", "score": 0.65},
        ],
    },
    {
        "snapshot_id": 3, "client": "codex", "model": "gpt-5",
        "first_seen": "2026-08-04T09:00:00", "last_seen": "2026-08-04T09:00:00",
        "session_count": 2,
        "hits": [{"kind": "tool", "title": "apply_patch", "text": "patch files", "score": 0.72}],
    },
]


def _build_html(tmp_path):
    inject = f"<script>window.__KB_TEST_GROUPS = {json.dumps(GROUPS)};</script>"
    path = tmp_path / "kb_dashboard.html"
    path.write_text(read_dashboard_template().replace("</head>", inject + "</head>", 1), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    from playwright.sync_api import sync_playwright

    html = _build_html(tmp_path_factory.mktemp("kb"))
    pw = sync_playwright().start()
    # Playwright's CDN browser download is blocked by the local DLP proxy; use installed Google Chrome
    browser = pw.chromium.launch(headless=True, channel="chrome")
    pg = browser.new_page()
    pg.goto(f"file://{html}")
    # KB section starts hidden (tab UI); reveal it so Playwright clicks see visible elements
    pg.evaluate("document.querySelector('#kb-view').classList.remove('hidden')")
    pg.evaluate("renderKbResults(window.__KB_TEST_GROUPS)")
    yield pg
    browser.close()
    pw.stop()


def test_same_client_model_folded_into_latest_card(page):
    # 3 groups, 2 share client+model -> 2 top-level cards, 1 history fold
    assert page.eval_on_selector_all("#kb-results > .kb-group", "els => els.length") == 2
    assert page.eval_on_selector_all("#kb-results > .kb-group .kb-history-fold", "els => els.length") == 1
    # latest card is the 2026-08-06 snapshot with the "latest" badge
    assert page.eval_on_selector_all("#kb-results > .kb-group .kb-latest-badge", "els => els.length") == 2
    summary = page.eval_on_selector(".kb-summary", "el => el.textContent")
    assert "2" in summary  # 2 result groups after folding


def test_score_bar_badge_and_tier_classes(page):
    assert page.eval_on_selector_all(".kb-score.score-high", "els => els.length") >= 1
    assert page.eval_on_selector_all(".kb-score.score-mid", "els => els.length") >= 1
    assert page.eval_on_selector_all(".kb-score.score-low", "els => els.length") >= 1
    assert page.eval_on_selector_all(".kb-hit-kind.tool", "els => els.length") >= 1
    assert page.eval_on_selector_all(".kb-hit-kind.prompt", "els => els.length") >= 1
    width = page.eval_on_selector(".kb-score-fill", "el => el.style.width")
    assert width.endswith("%") and width != "0%"


def test_expand_full_text_toggles(page):
    before = page.eval_on_selector(".kb-hit-text", "el => el.textContent.length")
    page.click(".kb-expand-btn")
    after = page.eval_on_selector(".kb-hit-text", "el => el.textContent.length")
    assert after > before
    assert page.eval_on_selector(".kb-hit-text", "el => el.classList.contains('full')")
    page.click(".kb-expand-btn")
    assert page.eval_on_selector(".kb-hit-text", "el => el.textContent.length") == before


def test_min_score_slider_filters_without_refetch(page):
    page.evaluate("""() => {
        const slider = document.querySelector('#kb-min-score');
        slider.value = '0.8';
        kbRenderFiltered();
    }""")
    # 0.95 and 0.823 survive; 0.65 / 0.72 filtered out -> codex card gone entirely
    titles = page.eval_on_selector_all(".kb-hit-title", "els => els.map(e => e.textContent)")
    assert "apply_patch" not in titles
    assert page.eval_on_selector_all("#kb-results > .kb-group", "els => els.length") == 1
    # summary reports the min-score filter kicked in (zh default lang)
    summary = page.eval_on_selector(".kb-summary", "el => el.textContent")
    assert "过滤" in summary
    page.evaluate("""() => {
        const slider = document.querySelector('#kb-min-score');
        slider.value = '0';
        kbRenderFiltered();
    }""")
    assert page.eval_on_selector_all("#kb-results > .kb-group", "els => els.length") == 2


def test_empty_state_shows_guidance(page):
    page.evaluate("renderKbResults([])")
    assert page.eval_on_selector_all(".kb-empty", "els => els.length") == 1
    hint = page.eval_on_selector(".kb-empty", "el => el.textContent")
    assert len(hint) > 10


MESSAGES = [
    {
        "session_id": "sess-abc", "client": "claude", "model": "k3-256k",
        "hits": [
            {"text": "how do I fix the race condition in the worker pool",
             "timestamp": "2026-08-09T10:00:00Z", "score": 0.87},
        ],
    },
]


def test_kb_message_section_rendered(page, tmp_path):
    html_path = tmp_path / "dashboard.html"
    html_path.write_text(read_dashboard_template(), encoding="utf-8")
    page.goto(f"file://{html_path}")
    # KB section starts hidden (tab UI); reveal it so inner_text sees rendered text
    page.evaluate("document.querySelector('#kb-view').classList.remove('hidden')")
    page.evaluate(
        """([groups, messages]) => {
            kbLastGroups = groups;
            renderKbResults(groups, messages);
        }""",
        [GROUPS, MESSAGES],
    )
    # default UI language is zh-CN
    section = page.locator(".kb-section-title", has_text="会话")
    assert section.count() == 1
    card = page.locator(".kb-message-group")
    assert card.count() == 1
    link = card.locator("a.kb-session-link")
    assert link.get_attribute("href") == "/dashboard/session/sess-abc"
    assert "race condition" in card.locator(".kb-hit-text").inner_text()
    assert card.locator(".kb-hit-kind.message").count() == 1
    # en dictionary renders the same section as "Sessions"
    page.evaluate("() => { setLang('en'); kbRenderFiltered(); }")
    assert page.locator(".kb-section-title", has_text="Sessions").count() == 1
