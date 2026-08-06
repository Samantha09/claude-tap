"""Deterministic bag-of-words hash embedder for tests (no model download)."""

from __future__ import annotations

import hashlib
import math
import re


class FakeEmbedder:
    name = "fake"
    dimension = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vec[digest[0] % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
