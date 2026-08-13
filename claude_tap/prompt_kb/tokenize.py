"""jieba-based segmentation for FTS keyword search over zh+en mixed text.

Both the FTS write path (store.py) and the query path (search.py) MUST use
segment() so index terms and query terms are cut identically.
"""

from __future__ import annotations

import warnings

_jieba = None
_jieba_failed = False


def _load_jieba():
    global _jieba, _jieba_failed
    if _jieba is not None or _jieba_failed:
        return _jieba
    try:
        import jieba
    except ImportError:
        _jieba_failed = True
        warnings.warn(
            "jieba is not installed; Chinese keyword search is degraded: pip install 'claude-tap[rag]'",
            stacklevel=2,
        )
        return None
    _jieba = jieba
    return _jieba


def segment(text: str) -> str:
    """Cut text into space-joined search tokens; raw text when jieba is missing."""
    jieba = _load_jieba()
    if jieba is None:
        return text
    return " ".join(jieba.cut_for_search(text))
