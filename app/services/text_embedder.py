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
        self._embedder = None  # triton_shared.models.embedder.TextEmbedder
        self._dim: int = 768  # embeddinggemma-300m output dimension

    async def _ensure_client(self) -> None:
        if self._embedder is not None:
            return
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

    async def embed(self, text: str) -> list[float]:
        """Return an L2-normalized embedding for *text*.

        Returns an empty list if text is None or empty.
        """
        if not text or not text.strip():
            return []
        await self._ensure_client()
        return await self._embedder.embed_query(text)

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
