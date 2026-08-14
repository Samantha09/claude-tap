"""End-to-end smoke test: real MCP client over stdio (skipped without [mcp])."""

import json
import os
import sys

import pytest

pytest.importorskip("mcp")  # requires the [mcp] extra

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


async def test_stdio_roundtrip(tmp_path):
    env = {**os.environ, "CLOUDTAP_DB": str(tmp_path / "traces.sqlite3")}
    params = StdioServerParameters(command=sys.executable, args=["-m", "claude_tap", "mcp"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {"kb_search", "kb_status"} <= {t.name for t in tools.tools}
            result = await session.call_tool("kb_status", {})
            # mcp 2.x renamed the wire field's Python attribute: isError → is_error.
            assert not getattr(result, "is_error", getattr(result, "isError", False))
            payload = json.loads(result.content[0].text)
            for key in ("snapshots", "chunks", "pending", "failed", "indexed", "messages", "embedder"):
                assert key in payload
