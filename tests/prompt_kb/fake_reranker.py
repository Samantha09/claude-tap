"""Deterministic lexical-overlap reranker for tests (no model download)."""

from __future__ import annotations

import re


class FakeReranker:
    name = "fake-reranker"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return [0.0] * len(texts)
        return [
            len(query_tokens & set(re.findall(r"[a-z0-9]+", text.lower()))) / len(query_tokens) for text in texts
        ]
