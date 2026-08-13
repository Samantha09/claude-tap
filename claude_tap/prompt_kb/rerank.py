"""Cross-encoder reranker: rescore fused candidates for calibrated ordering.

The reranker is an enhancement, never a requirement: every load/runtime
failure surfaces as RerankerUnavailable so callers can fall back to the
fused ranking (search reports reranked=false in that case).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from claude_tap.prompt_kb.embed import DEFAULT_RERANKER_MODEL

if TYPE_CHECKING:
    from claude_tap.prompt_kb.embed import KbConfig


class RerankerUnavailable(Exception):
    """The reranker model cannot be loaded."""


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class LocalReranker:
    """Local bge cross-encoder; predict() logits squashed through sigmoid."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerUnavailable(
                "sentence-transformers is not installed; install the optional dependency: pip install 'claude-tap[rag]'"
            ) from exc
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:
            # Download/load failures (network, TLS, disk, corrupt cache) must
            # degrade to the fused ranking, not crash the search path.
            raise RerankerUnavailable(f"failed to load reranker model {model_name!r}: {exc}") from exc
        self.name = f"reranker:{model_name}"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = self._model.predict([[query, text] for text in texts])
        return [1.0 / (1.0 + math.exp(-float(logit))) for logit in logits]


def create_reranker(config: KbConfig) -> Reranker | None:
    """None when configured off; raises RerankerUnavailable on load failure."""
    if config.reranker == "off":
        return None
    return LocalReranker(config.reranker_model)


def reranker_status(config: KbConfig) -> str:
    """Human-readable state for kb_status/stats: off | unavailable | model name."""
    if config.reranker == "off":
        return "off"
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "unavailable"
    return config.reranker_model
