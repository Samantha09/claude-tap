"""User-message extraction from trace records across provider formats."""

import hashlib

import pytest

from claude_tap.prompt_kb.messages import (
    MIN_ASSISTANT_CHARS,
    _keep_text,
    extract_assistant_messages,
    extract_user_messages,
    message_content_hash,
)


def _record(body, path="/v1/messages", timestamp="2026-08-10T01:00:00Z"):
    return {
        "timestamp": timestamp,
        "request": {"method": "POST", "path": path, "body": body},
        "response": {"status": 200},
    }


def test_anthropic_user_messages():
    records = [
        _record(
            {
                "model": "k3",
                "messages": [
                    {"role": "user", "content": "how do I fix the race condition"},
                    {"role": "assistant", "content": "use a lock"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "it still hangs"},
                            {"type": "image", "source": {"data": "..."}},
                        ],
                    },
                ],
            }
        ),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == [
        "how do I fix the race condition",
        "it still hangs",
    ]
    assert msgs[0].record_index == 0
    assert msgs[1].message_index == 1
    assert msgs[0].timestamp == "2026-08-10T01:00:00Z"


def test_anthropic_tool_result_blocks_skipped():
    records = [
        _record(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t1", "content": "file contents here"},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t2", "content": "output"},
                            {"type": "text", "text": "now fix it"},
                        ],
                    },
                ],
            }
        ),
    ]
    msgs = extract_user_messages(records)
    # First message is pure tool_result -> dropped; second keeps only text part
    assert [m.text for m in msgs] == ["now fix it"]


def test_openai_chat_completions():
    records = [
        _record(
            {
                "model": "gpt-5",
                "messages": [
                    {"role": "system", "content": "dev"},
                    {"role": "user", "content": "refactor the parser"},
                    {"role": "tool", "tool_call_id": "c1", "content": "result"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "and add tests"},
                        ],
                    },
                ],
            },
            path="/v1/chat/completions",
        ),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["refactor the parser", "and add tests"]


def test_openai_responses_input():
    records = [
        _record(
            {
                "model": "gpt-5",
                "instructions": "dev",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "explain this repo"}],
                    },
                    {"type": "function_call_output", "call_id": "c1", "output": "x"},
                ],
            },
            path="/v1/responses",
        ),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["explain this repo"]


def test_gemini_contents():
    records = [
        _record(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "write a haiku"}]},
                    {
                        "role": "user",
                        "parts": [
                            {"functionResponse": {"name": "f", "response": {}}},
                        ],
                    },
                    {"role": "model", "parts": [{"text": "ok"}]},
                ],
            },
            path="/v1beta/models/gemini-3:generateContent",
        ),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["write a haiku"]


def test_harness_injected_messages_filtered():
    records = [
        _record(
            {
                "messages": [
                    {"role": "user", "content": "<system-reminder>secret</system-reminder>"},
                    {"role": "user", "content": "<command-message>/clear</command-message>"},
                    {"role": "user", "content": "<local-command-stdout>out</local-command-stdout>"},
                    {"role": "user", "content": "   "},
                    {"role": "user", "content": "real question"},
                ],
            }
        ),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["real question"]


@pytest.mark.parametrize(
    "text",
    [
        # First XML tag in viewer drop set
        "<environment_context>cwd=/x, date=2026-08-10</environment_context>",
        "<user_information>name: san</user_information>",
        "<skills><skill>a</skill></skills>",
        "<artifacts>artifact list</artifacts>",
        "<codex_internal_context>ctx</codex_internal_context>",
        "<local-command-caveat>caveat</local-command-caveat>",
        "<session_context>ctx</session_context>",
        "<slash_commands>/a /b</slash_commands>",
        "<subagents>agents</subagents>",
        "<system-reminder>secret</system-reminder>",
        # startswith drops
        "<INSTRUCTIONS>do not break things</INSTRUCTIONS>",
        "# AGENTS.md instructions\nFollow the rules in AGENTS.md",
        "# Files mentioned by the user:\n- foo.py",
        # regex drops
        "<image_input>",
        "</image>",
        "<image width=100>",
        "[SUGGESTION MODE: rewrite the last reply]",
        "web page content: hello world",
        "Page Content: hello world",
        "网页内容:你好",
        "[Image: source: /path/x.png]",
    ],
)
def test_viewer_drop_rules_applied(text):
    assert _keep_text(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "how do I fix the race condition",
        "这个接口为什么会超时?",
        "[Image #1] 这张图里的报错是什么原因?",
        "please review <session_context> block inside my text",
        "use the <skills> directory layout",
        "my web page content pipeline is broken",
    ],
)
def test_genuine_user_text_kept(text):
    assert _keep_text(text) is True


def test_viewer_drop_rules_end_to_end():
    records = [
        _record(
            {
                "messages": [
                    {"role": "user", "content": "<environment_context>cwd=/x</environment_context>"},
                    {"role": "user", "content": "# AGENTS.md instructions\nrules here"},
                    {"role": "user", "content": "[SUGGESTION MODE: ...]"},
                    {"role": "user", "content": "网页内容:抓取结果"},
                    {"role": "user", "content": "[Image: source: /tmp/a.png]"},
                    {"role": "user", "content": "<image_input>"},
                    {"role": "user", "content": "actual user question"},
                ],
            }
        ),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["actual user question"]


def test_long_message_split():
    long_text = ("paragraph one. " * 200) + "\n\n" + ("paragraph two. " * 200)
    records = [_record({"messages": [{"role": "user", "content": long_text}]})]
    msgs = extract_user_messages(records)
    assert len(msgs) > 1
    assert all(len(m.text) <= 2000 for m in msgs)
    # split pieces share record_index, message_index increments
    assert msgs[0].record_index == msgs[1].record_index


def test_content_hash_normalizes():
    h1 = message_content_hash("hello world  \n")
    h2 = message_content_hash("hello world")
    assert h1 == h2
    assert h1 == hashlib.sha256(b"hello world").hexdigest()


def test_unknown_provider_and_empty_body_skipped():
    records = [
        {"timestamp": "t", "request": {"path": "/health", "body": {}}},
        _record({"messages": [{"role": "assistant", "content": "hi"}]}),
    ]
    assert extract_user_messages(records) == []


def _assistant_record(resp_body, path="/v1/messages", timestamp="2026-08-10T01:00:00Z"):
    return {
        "timestamp": timestamp,
        "request": {"method": "POST", "path": path, "body": {}},
        "response": {"status": 200, "body": resp_body},
    }


LONG = "this is a sufficiently long assistant reply explaining the fix"


def test_anthropic_assistant_text_only():
    records = [
        _assistant_record(
            {
                "content": [
                    {"type": "thinking", "thinking": "let me think about this problem"},
                    {"type": "text", "text": LONG},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ]
            }
        )
    ]
    msgs = extract_assistant_messages(records)
    assert [m.text for m in msgs] == [LONG]
    assert msgs[0].record_index == 0 and msgs[0].message_index == 0
    assert msgs[0].timestamp == "2026-08-10T01:00:00Z"


def test_openai_chat_assistant_text():
    records = [
        _assistant_record(
            {"choices": [{"message": {"role": "assistant", "content": LONG}}]},
            path="/v1/chat/completions",
        )
    ]
    assert [m.text for m in extract_assistant_messages(records)] == [LONG]


def test_openai_responses_assistant_text():
    records = [
        _assistant_record(
            {
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": LONG}],
                    },
                    {"type": "function_call", "name": "shell", "arguments": "{}"},
                ]
            },
            path="/v1/responses",
        )
    ]
    assert [m.text for m in extract_assistant_messages(records)] == [LONG]


def test_gemini_assistant_text_skips_thought_parts():
    records = [
        _assistant_record(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "hidden reasoning", "thought": True},
                                {"text": LONG},
                            ]
                        }
                    }
                ]
            },
            path="/v1beta/models/gemini-2.0-flash:generateContent",
        )
    ]
    assert [m.text for m in extract_assistant_messages(records)] == [LONG]


def test_short_and_empty_replies_dropped():
    records = [
        _assistant_record({"content": [{"type": "text", "text": "好的"}]}),
        _assistant_record({"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}),
        {"timestamp": "t", "request": {"path": "/v1/messages"}, "response": {"status": 200}},  # no body
    ]
    assert extract_assistant_messages(records) == []
    assert MIN_ASSISTANT_CHARS == 20


def test_long_reply_split_into_pieces():
    piece = "paragraph with enough words to pass the minimum length filter. "
    records = [_assistant_record({"content": [{"type": "text", "text": (piece * 60)}]})]
    msgs = extract_assistant_messages(records)
    assert len(msgs) > 1
    assert [m.message_index for m in msgs] == list(range(len(msgs)))
