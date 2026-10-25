"""BGE-M3 Dense Embedding backed by Sentence Transformers."""

import math
from collections.abc import Sequence
from typing import Any

from sentence_transformers import SentenceTransformer

from evidenceflow.model_device import resolve_model_device


class EmbeddingError(RuntimeError):
    """Common base exception for embedding model load, execution, or output-conversion failures."""


class EmbeddingLoadError(EmbeddingError):
    """Signals that the Sentence Transformer model cannot be loaded."""


class EmbeddingNormalizationError(EmbeddingError):
    """Signals that a model vector cannot be L2-normalized safely."""


def _normalize_vector(vector: Any) -> list[float]:
    """Convert a third-party vector into finite Python floats and L2-normalize it explicitly."""

    # A NumPy array offers tolist(); a test Fake may return list/tuple directly.
    raw_values = vector.tolist() if hasattr(vector, "tolist") else vector
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
        raise EmbeddingNormalizationError("Embedding output is not a vector")
    if not raw_values:
        raise EmbeddingNormalizationError("Embedding vector must not be empty")

    try:
        # Convert to builtin float at the Provider boundary so numpy.float32 never leaks into the
        # business layer.
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError) as exc:
        raise EmbeddingNormalizationError(
            "Embedding vector contains non-numeric values"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingNormalizationError(
            "Embedding vector contains non-finite values"
        )

    norm = math.sqrt(math.fsum(value * value for value in values))
    # A zero vector has no direction and must fail loudly, rather than producing NaN that would
    # later be written into a Vector Store.
    if norm == 0.0 or not math.isfinite(norm):
        raise EmbeddingNormalizationError(
            "Embedding vector has an invalid L2 norm"
        )
    return [float(value / norm) for value in values]


def _normalize_batch(raw_vectors: Any, expected_count: int) -> list[list[float]]:
    """Validate the batch output count, then convert and normalize each model vector."""

    vectors = raw_vectors.tolist() if hasattr(raw_vectors, "tolist") else raw_vectors
    if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
        raise EmbeddingNormalizationError("Embedding batch output is invalid")
    if len(vectors) != expected_count:
        raise EmbeddingNormalizationError(
            "Embedding batch size does not match input text count"
        )
    return [_normalize_vector(vector) for vector in vectors]


def _validate_text(text: str, *, field_name: str) -> None:
    """Reject non-string and whitespace-only input without altering the original text."""

    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be a string")
    if not text.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace")


class BgeM3EmbeddingProvider:
    """Load the configured BGE-M3 model once and produce normalized Dense Vectors."""

    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        """Initialize and cache the model instance so documents and queries later share one vector space."""

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Embedding model name must not be empty")
        self.model_name = model_name
        self.device = resolve_model_device(device)
        try:
            # Documents and queries share this instance and device; the model library loads float32
            # weights by default.
            self._model = SentenceTransformer(model_name, device=self.device)
        except Exception as exc:
            raise EmbeddingLoadError(
                f"Failed to load embedding model: {model_name}"
            ) from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode several document texts in one batch, guaranteeing every output is explicitly L2-normalized."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        text_list = list(texts)
        if not text_list:
            # An empty batch returns immediately without invoking the expensive model inference.
            return []
        for index, text in enumerate(text_list):
            _validate_text(text, field_name=f"texts[{index}]")

        try:
            # The underlying call receives the whole Batch at once; the project layer then
            # independently guarantees normalization and output types.
            raw_vectors = self._model.encode(
                text_list,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
                device=self.device,
            )
        except Exception as exc:
            raise EmbeddingError("Failed to embed document texts") from exc
        return _normalize_batch(raw_vectors, expected_count=len(text_list))

    def embed_query(self, query: str) -> list[float]:
        """Produce a single normalized query vector using the same model as documents."""

        _validate_text(query, field_name="query")
        try:
            # No strip, rewrite, or translation: the validated raw Query goes to the model
            # unchanged.
            raw_vector = self._model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
                device=self.device,
            )
        except Exception as exc:
            raise EmbeddingError("Failed to embed query") from exc
        return _normalize_vector(raw_vector)
