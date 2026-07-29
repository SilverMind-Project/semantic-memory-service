"""Text embedding service for scene description semantic search.

Uses embeddinggemma-300m served by Triton Inference Server via the
triton-shared library to generate 768-dimensional text embeddings.

Design
------
``TextEmbedder`` is the ABC. ``TritonTextEmbedder`` wraps
triton-shared's TextEmbedder with lazy client initialization.
``NullTextEmbedder`` returns an empty list and is used when text
embedding is disabled.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # triton_shared is an optional runtime dependency, imported lazily inside
    # _ensure_client() so the service starts without it. These names exist for
    # annotations only and are never imported at runtime.
    from triton_shared.client.grpc import TritonGrpcClient
    from triton_shared.models.embedder import TextEmbedder as _TritonEmbedder

logger = logging.getLogger(__name__)


class TextEmbedder(ABC):
    """Abstract text embedder."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return a normalized embedding vector for *text*.

        Returns an empty list when the model is unavailable or text is empty.
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True when the model is loaded and ready."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of the embedding vectors produced by this model."""
        ...

    async def aclose(self) -> None:
        """Release any upstream connection. No-op unless an embedder holds one."""
        return None


class NullTextEmbedder(TextEmbedder):
    """No-op embedder for graceful degradation."""

    async def embed(self, text: str) -> list[float]:
        return []

    @property
    def is_available(self) -> bool:
        return False

    @property
    def embedding_dim(self) -> int:
        return 0


class TritonTextEmbedder(TextEmbedder):
    """Text embedder via Triton Inference Server.

    Uses embeddinggemma-300m which produces 768-dimensional
    L2-normalized embeddings suitable for cosine similarity search.
    The Triton gRPC client is created lazily on first use.

    Args:
        triton_url: Triton gRPC endpoint (e.g. ``localhost:8701``).
        model_name: Triton model name.
        tokenizer_path: Path to the HuggingFace ``tokenizer.json`` file.
    """

    def __init__(
        self,
        triton_url: str,
        model_name: str,
        tokenizer_path: str,
    ) -> None:
        self._triton_url = triton_url
        self._model_name = model_name
        self._tokenizer_path = tokenizer_path
        self._embedder: _TritonEmbedder | None = None
        self._grpc_client: TritonGrpcClient | None = None
        self._dim: int = 768  # embeddinggemma-300m output dimension

    async def _ensure_client(self) -> _TritonEmbedder:
        """Return the lazily built embedder, connecting to Triton on first use.

        Returns the embedder rather than ``None`` so callers get a non-optional
        value; assigning to ``self._embedder`` alone leaves every call site
        needing to re-narrow the attribute.
        """
        if self._embedder is not None:
            return self._embedder
        from triton_shared.client.grpc import TritonGrpcClient
        from triton_shared.models.embedder import TextEmbedder as _TritonEmbedder

        logger.info(
            "connecting to Triton url=%s model=%s",
            self._triton_url,
            self._model_name,
        )
        self._grpc_client = TritonGrpcClient(self._triton_url)
        await self._grpc_client.__aenter__()
        self._embedder = _TritonEmbedder(
            client=self._grpc_client,
            model_name=self._model_name,
            tokenizer_path=self._tokenizer_path,
        )
        logger.info("Triton text embedder ready dim=%d", self._dim)
        return self._embedder

    async def embed(self, text: str) -> list[float]:
        """Return an L2-normalized embedding for *text*.

        Returns an empty list if text is None or empty.
        """
        if not text or not text.strip():
            return []
        embedder = await self._ensure_client()
        result: list[float] = await embedder.embed_query(text)
        return result

    async def aclose(self) -> None:
        """Release the Triton gRPC connection opened by ``_ensure_client``.

        ``_ensure_client`` enters the client's async context manager but nothing
        exited it, so the connection outlived the process's use of it. Safe to
        call when the client was never built.
        """
        if self._grpc_client is None:
            return
        client: Any = self._grpc_client
        self._grpc_client = None
        self._embedder = None
        try:
            await client.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.warning("Triton gRPC client close failed", exc_info=True)

    @property
    def is_available(self) -> bool:
        return True

    @property
    def embedding_dim(self) -> int:
        return self._dim


def build_text_embedder(
    *,
    enabled: bool,
    triton_url: str,
    model_name: str,
    tokenizer_path: str,
) -> TextEmbedder:
    """Construct a TextEmbedder from config values.

    Args:
        enabled: Whether text embedding is enabled.
        triton_url: Triton gRPC endpoint.
        model_name: Triton model name.
        tokenizer_path: Path to tokenizer.json.

    Returns:
        A TextEmbedder instance (either TritonTextEmbedder or NullTextEmbedder).
    """
    if not enabled:
        logger.info("text_embedding_disabled returning_null_embedder")
        return NullTextEmbedder()

    return TritonTextEmbedder(
        triton_url=triton_url,
        model_name=model_name,
        tokenizer_path=tokenizer_path,
    )
