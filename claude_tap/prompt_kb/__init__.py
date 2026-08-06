"""Local prompt/tool knowledge base over captured trace snapshots."""

from claude_tap.prompt_kb.index import index_pending, rebuild_index, run_index_loop
from claude_tap.prompt_kb.search import search
from claude_tap.prompt_kb.store import KbStore

__all__ = ["KbStore", "index_pending", "rebuild_index", "run_index_loop", "search"]
