"""Embedder abstraction: local sentence-transformers by default, API optional.

Configuration comes from `[prompt_kb]` in the config TOML (default path
`~/.config/claude-tap/config.toml`), overridable per-key with
`CLAUDE_TAP_KB_*` environment variables.
"""

from __future__ import annotations

import array
import json
import os
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "claude-tap" / "config.toml"


class EmbedderUnavailable(Exception):
    """Raised when no usable embedder is configured or installed."""


class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class KbConfig:
    embedder: str = "local"  # "local" | "api"
    local_model: str = DEFAULT_LOCAL_MODEL
    api_base: str = ""
    api_model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    # E5-family models require asymmetric prefixes; without them scores
    # compress into a narrow band and irrelevant text scores deceptively high.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


def load_config(path: Path | None = None) -> KbConfig:
    values: dict[str, str] = {}
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.is_file():
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        section = data.get("prompt_kb") or {}
        values.update({k: str(v) for k, v in section.items()})
    for key in (
        "embedder",
        "local_model",
        "api_base",
        "api_model",
        "api_key_env",
        "query_prefix",
        "passage_prefix",
    ):
        env = os.environ.get(f"CLAUDE_TAP_KB_{key.upper()}")
        if env:
            values[key] = env
    known = {f for f in KbConfig.__dataclass_fields__}
    return KbConfig(**{k: v for k, v in values.items() if k in known})


class LocalEmbedder:
    """Local sentence-transformers embedder, tuned for the E5 family.

    Documents go through embed() with the passage prefix, queries through
    embed_query() with the query prefix. The "+e5p" name marker distinguishes
    prefixed vector spaces from pre-prefix indexes (kb_meta mismatch → reindex).
    """

    def __init__(
        self, model_name: str = DEFAULT_LOCAL_MODEL, *, query_prefix: str = "query: ", passage_prefix: str = "passage: "
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderUnavailable(
                "sentence-transformers is not installed; install the optional dependency: pip install 'claude-tap[rag]'"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
            get_dim = (
                getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            )
            self.dimension = int(get_dim())
        except Exception as exc:
            # Model download/load failures (network, TLS, disk, corrupt cache)
            # must surface as EmbedderUnavailable so callers return the 501
            # embedder_unavailable hint instead of a bare 500.
            raise EmbedderUnavailable(f"failed to load local embedding model {model_name!r}: {exc}") from exc
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        marker = "+e5p" if (query_prefix or passage_prefix) else ""
        self.name = f"local:{model_name}{marker}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode([self._passage_prefix + t for t in texts], normalize_embeddings=True)
        return [list(map(float, vec)) for vec in vectors]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode([self._query_prefix + t for t in texts], normalize_embeddings=True)
        return [list(map(float, vec)) for vec in vectors]


class ApiEmbedder:
    """OpenAI-compatible /embeddings client over stdlib urllib."""

    def __init__(self, *, api_base: str, api_model: str, api_key: str):
        self._endpoint = api_base.rstrip("/") + "/embeddings"
        self._model = api_model
        self._api_key = api_key
        self.name = f"api:{api_model}"
        self.dimension = 0  # unknown until first response

    def embed(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps({"model": self._model, "input": texts}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        vectors = [list(map(float, item["embedding"])) for item in payload["data"]]
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        # OpenAI-compatible APIs have no query/passage prefix convention.
        return self.embed(texts)


def create_embedder(config: KbConfig) -> Embedder:
    if config.embedder == "api":
        if not config.api_base or not config.api_model:
            raise EmbedderUnavailable("api embedder requires api_base and api_model")
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise EmbedderUnavailable(f"api embedder requires the {config.api_key_env} environment variable")
        return ApiEmbedder(api_base=config.api_base, api_model=config.api_model, api_key=api_key)
    return LocalEmbedder(
        config.local_model,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
    )


def vectors_to_blob(vectors: list[list[float]]) -> list[bytes]:
    blobs = []
    for vec in vectors:
        arr = array.array("f", vec)
        blobs.append(arr.tobytes())
    return blobs
