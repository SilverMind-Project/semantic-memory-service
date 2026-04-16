"""Text embedding service for scene description semantic search.

Uses sentence-transformers with all-MiniLM-L6-v2 model to generate
384-dimensional text embeddings for semantic search of scene descriptions.

Design
------
``TextEmbedder`` is the ABC. ``SentenceTransformerEmbedder`` wraps
sentence-transformers. ``NullTextEmbedder`` returns an empty list and is
used when text embedding is disabled or sentence-transformers is not installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class TextEmbedder(ABC):
    """Abstract text embedder."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
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

    def embed(self, text: str) -> List[float]:
        return []

    @property
    def is_available(self) -> bool:
        return False

    @property
    def embedding_dim(self) -> int:
        return 0


class SentenceTransformerEmbedder(TextEmbedder):
    """Text embedder via sentence-transformers.

    Uses all-MiniLM-L6-v2 model which produces 384-dimensional
    L2-normalized embeddings suitable for cosine similarity search.

    Args:
        model_name: sentence-transformers model ID (default "all-MiniLM-L6-v2").
        device: PyTorch device string (default "cpu").
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers and torch are required for "
                "SentenceTransformerEmbedder. "
                "Install with: pip install sentence-transformers torch"
            ) from exc

        logger.info(
            "loading_sentence_transformers model=%s device=%s",
            model_name,
            device,
        )
        self._device = device
        self._model = SentenceTransformer(model_name, device=device)
        self._torch = torch
        self._dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "sentence_transformers_loaded model=%s dim=%d device=%s",
            model_name,
            self._dim,
            device,
        )

    def embed(self, text: str) -> List[float]:
        """Return an L2-normalized embedding for *text*.

        Returns an empty list if text is None or empty.
        """
        if not text or not text.strip():
            return []

        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embedding[0].cpu().float().tolist()

    @property
    def is_available(self) -> bool:
        return True

    @property
    def embedding_dim(self) -> int:
        return self._dim


def build_text_embedder(
    *,
    enabled: bool,
    model_name: str,
    device: str,
) -> TextEmbedder:
    """Construct a TextEmbedder from config values.

    Args:
        enabled: Whether text embedding is enabled.
        model_name: sentence-transformers model ID.
        device: PyTorch device string.

    Returns:
        A TextEmbedder instance (either SentenceTransformerEmbedder or NullTextEmbedder).
    """
    if not enabled:
        logger.info("text_embedding_disabled returning_null_embedder")
        return NullTextEmbedder()

    try:
        return SentenceTransformerEmbedder(
            model_name=model_name,
            device=device,
        )
    except RuntimeError as exc:
        logger.warning(
            "sentence_transformers_not_installed_or_failed: %s returning_null_embedder",
            exc,
        )
        return NullTextEmbedder()
