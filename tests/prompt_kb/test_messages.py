"""User-message extraction from trace records across provider formats."""

import hashlib

from claude_tap.prompt_kb.messages import (
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
                    {"role": "user", "content": "   "},
                    {"role": "user", "content": "real question"},
                ],
            }
        ),
    ]
    msgs = extract_user_messages(records)
    assert [m.text for m in msgs] == ["real question"]


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
