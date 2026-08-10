"""Lazy indexing: embed pending chunks in a long-lived dashboard thread."""

from __future__ import annotations

import logging
import threading

from claude_tap.prompt_kb.embed import Embedder, EmbedderUnavailable, create_embedder, load_config, vectors_to_blob
from claude_tap.prompt_kb.extract import extract_unprocessed
from claude_tap.prompt_kb.store import KbStore
from claude_tap.trace_store import get_trace_store

logger = logging.getLogger(__name__)


def index_pending(store: KbStore, embedder: Embedder, *, batch_size: int = 32) -> dict:
    indexed = failed = 0
    meta_synced = False
    # Retry chunks that failed in earlier rounds (e.g. transient embedder
    # errors); chunks that hit the attempt cap stay failed.
    store.requeue_failed()
    while True:
        batch = store.pending_chunks(batch_size)
        if not batch:
            break
        try:
            vectors = embedder.embed([row["text"] for row in batch])
        except Exception as exc:  # noqa: BLE001 - one bad batch must not stop indexing
            logger.warning("embedding batch failed: %s", exc)
            for row in batch:
                store.mark_chunk_failed(row["id"])
            failed += len(batch)
            continue
        if not meta_synced:
            meta_synced = True
            if embedder.dimension:
                # ApiEmbedder only learns its dimension from the first response;
                # persist it now so kb_meta never keeps embedding_dim="0".
                ensure_embedder_meta(store, embedder)
        for row, blob in zip(batch, vectors_to_blob(vectors)):
            store.mark_chunk_indexed(row["id"], blob)
        indexed += len(batch)
    messages_indexed, messages_failed = _index_pending_messages(store, embedder, batch_size)
    return {
        "indexed": indexed,
        "failed": failed,
        "messages_indexed": messages_indexed,
        "messages_failed": messages_failed,
        "remaining": store.stats()["pending"],
    }


def _index_pending_messages(store: KbStore, embedder: Embedder, batch_size: int) -> tuple[int, int]:
    """Embed pending user messages; same retry semantics as chunk indexing."""
    indexed = failed = 0
    store.requeue_failed_messages()
    while True:
        batch = store.pending_messages(batch_size)
        if not batch:
            break
        try:
            vectors = embedder.embed([row["text"] for row in batch])
        except Exception:  # noqa: BLE001 - one bad batch must not stop indexing
            logger.warning("message embedding batch failed", exc_info=True)
            for row in batch:
                store.mark_message_failed(row["id"])
            failed += len(batch)
            continue
        for row, blob in zip(batch, vectors_to_blob(vectors)):
            store.mark_message_indexed(row["id"], blob)
        indexed += len(batch)
    return indexed, failed


def rebuild_index(store: KbStore, embedder: Embedder) -> dict:
    store.reset_embeddings()
    store.reset_message_embeddings()
    return index_pending(store, embedder)


def ensure_embedder_meta(store: KbStore, embedder: Embedder) -> None:
    store.set_meta("embedder_name", embedder.name)
    store.set_meta("embedding_dim", str(embedder.dimension))


def run_index_loop(*, interval_seconds: float = 30.0, stop_event: threading.Event | None = None) -> None:
    """Background entry point for the dashboard process.

    Owns its own KbStore/TraceStore connections. Embedder creation is retried
    every 10 rounds when unavailable (e.g. model not downloaded yet).
    """
    stop = stop_event or threading.Event()
    store = KbStore.default()
    embedder: Embedder | None = None
    rounds_since_embedder_failure = 0
    while not stop.is_set():
        try:
            extract_unprocessed(store, get_trace_store())
            if embedder is None:
                if rounds_since_embedder_failure % 10 == 0:
                    try:
                        embedder = create_embedder(load_config())
                        ensure_embedder_meta(store, embedder)
                    except EmbedderUnavailable as exc:
                        logger.info("kb embedder unavailable: %s", exc)
                rounds_since_embedder_failure += 1
            if embedder is not None:
                index_pending(store, embedder)
        except Exception:  # noqa: BLE001 - the loop must never die silently
            logger.exception("kb index loop iteration failed")
        stop.wait(interval_seconds)
