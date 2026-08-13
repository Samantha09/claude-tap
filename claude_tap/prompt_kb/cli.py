"""`claude-tap kb` subcommand: search / reindex / status for the prompt KB."""

from __future__ import annotations

import argparse
import sys

from claude_tap.prompt_kb.embed import EmbedderUnavailable, create_embedder, load_config
from claude_tap.prompt_kb.index import ensure_embedder_meta, rebuild_index
from claude_tap.prompt_kb.rerank import RerankerUnavailable, create_reranker, reranker_status
from claude_tap.prompt_kb.search import ReindexRequired, search, search_messages
from claude_tap.prompt_kb.store import KbStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claude-tap kb")
    sub = parser.add_subparsers(dest="command", required=True)
    search_parser = sub.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--client")
    search_parser.add_argument("--kind", choices=["tool", "prompt_section"])
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--rel-delta", type=float, default=0.05)
    sub.add_parser("reindex")
    sub.add_parser("status")
    return parser


def _embedder_or_exit():
    try:
        return create_embedder(load_config())
    except EmbedderUnavailable as exc:
        print(f"kb embedder unavailable: {exc}", file=sys.stderr)
        return None


def _reranker_or_none():
    try:
        return create_reranker(load_config())
    except RerankerUnavailable:
        return None


def kb_main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    store = KbStore.default()
    if args.command == "status":
        stats = store.stats()
        print(" ".join(f"{key}={value}" for key, value in stats.items()))
        print(f"embedder={store.get_meta('embedder_name') or 'none'}")
        print(f"reranker={reranker_status(load_config())}")
        return 0
    embedder = _embedder_or_exit()
    if embedder is None:
        return 2
    if args.command == "reindex":
        ensure_embedder_meta(store, embedder)
        result = rebuild_index(store, embedder)
        print(
            f"indexed={result['indexed']} failed={result['failed']}"
            f" messages_indexed={result['messages_indexed']} messages_failed={result['messages_failed']}"
        )
        return 0
    reranker = _reranker_or_none()
    try:
        results, chunks_reranked = search(
            store, embedder, args.query, client=args.client, kind=args.kind, limit=args.limit,
            rel_delta=args.rel_delta, reranker=reranker,
        )
    except ReindexRequired as exc:
        print(str(exc), file=sys.stderr)
        return 3
    message_results, messages_reranked = search_messages(
        store, embedder, args.query, client=args.client, limit=args.limit,
        rel_delta=args.rel_delta, reranker=reranker,
    )
    print(f"reranked: {'yes' if chunks_reranked and messages_reranked else 'no'}")
    if message_results:
        print("messages:")
        for rank, group in enumerate(message_results, 1):
            print(f"[{rank}] session {group.session_id} ({group.client} / {group.model})")
            for hit in group.hits:
                print(f"    [{hit.role}] score={hit.score:.3f} {hit.timestamp}")
                print(f"    {hit.text[:200]}")
    for rank, group in enumerate(results, 1):
        print(
            f"[{rank}] {group.client} / {group.model} (first seen {group.first_seen}, sessions {group.session_count})"
        )
        for hit in group.hits:
            print(f"    {hit.kind} {hit.title} score={hit.score:.3f}")
            print(f"    {hit.text[:200]}")
    return 0
