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
