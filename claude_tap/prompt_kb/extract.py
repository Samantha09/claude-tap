"""Extract prompt snapshots from trace sessions into the KB store."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from claude_tap.prompt_kb.chunk import chunk_snapshot, content_hash
from claude_tap.prompt_kb.store import KbStore
from claude_tap.prompt_snapshot import snapshot_from_records
from claude_tap.trace_store import TraceStore

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_session(
    store: KbStore, *, session_id: str, client: str, records: list[dict[str, Any]], processed_at: str
) -> int | None:
    try:
        snapshot = snapshot_from_records(records)
    except (ValueError, KeyError, TypeError) as exc:
        logger.info("no prompt snapshot in session %s: %s", session_id, exc)
        store.record_source(session_id, None, processed_at)
        return None
    tools_json = json.dumps(
        [{"name": t.name, "description": t.description, "schema": t.schema} for t in snapshot.tools],
        ensure_ascii=False,
    )
    snapshot_id, created = store.upsert_snapshot(
        content_hash=content_hash(client, snapshot.model, snapshot),
        client=client,
        provider=snapshot.provider,
        model=snapshot.model,
        system_prompt=snapshot.system_prompt,
        developer_prompt=snapshot.developer_prompt,
        tools_json=tools_json,
        seen_at=snapshot.captured_at or processed_at,
    )
    if created:
        chunks = chunk_snapshot(snapshot)
        store.replace_chunks(snapshot_id, [(c.kind, c.title, c.text) for c in chunks])
    store.record_source(session_id, snapshot_id, processed_at)
    return snapshot_id


def extract_unprocessed(store: KbStore, trace: TraceStore, *, limit: int = 50) -> dict:
    processed = snapshots = skipped = 0
    rows = trace.list_session_rows(limit=limit * 4)  # over-fetch: some are filtered below
    for row in rows:
        if processed >= limit:
            break
        session_id = str(row["id"])
        if store.is_source_processed(session_id):
            continue
        if not row["record_count"]:
            store.record_source(session_id, None, _now())
            skipped += 1
            continue
        try:
            records = trace.load_records(session_id)
            snap_id = extract_session(
                store,
                session_id=session_id,
                client=str(row["client"] or "unknown"),
                records=records,
                processed_at=_now(),
            )
        except Exception as exc:
            # Do not record_source: the session stays retriable on the next pass.
            logger.warning("kb extraction failed for session %s: %s", session_id, exc)
            skipped += 1
            continue
        processed += 1
        if snap_id is not None:
            snapshots += 1
    return {"processed": processed, "snapshots": snapshots, "skipped": skipped}
