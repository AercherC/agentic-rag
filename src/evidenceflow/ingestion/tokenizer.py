"""Provide the Chunker with a Tokenizer interface decoupled from the Hugging Face implementation."""

import threading
from typing import Protocol, runtime_checkable

from transformers import AutoTokenizer


@runtime_checkable
class TokenizerProtocol(Protocol):
    """The minimal Tokenizer contract the Chunker needs."""

    def encode(self, text: str) -> list[int]:
        """Encode a body into an ID sequence that contains no model special tokens."""

        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode body Token IDs back into text."""

        ...


class TokenizerError(RuntimeError):
    """Common base exception for Tokenizer load, encode, or decode failures."""


class TokenizerLoadError(TokenizerError):
    """Signals that the Hugging Face Tokenizer cannot be loaded."""


class BgeM3Tokenizer:
    """Load a BGE-M3-compatible Hugging Face Tokenizer from the configured model name."""

    def __init__(self, model_name: str) -> None:
        """Load only the Tokenizer files; do not load model weights or initialize a GPU."""

        if not model_name.strip():
            raise ValueError("Tokenizer model name must not be empty")
        self.model_name = model_name
        # The same instance is shared across concurrent requests (the ContextBuilder counts tokens);
        # fast tokenizers are not safe for concurrent calls, so calls are serialized.
        self._lock = threading.Lock()
        try:
            # The model name is injected via Settings.embedding_model so that the Adapter does not
            # hardcode it twice.
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise TokenizerLoadError(
                f"Failed to load tokenizer for model: {model_name}"
            ) from exc

    def encode(self, text: str) -> list[int]:
        """Encode text into body Token IDs without counting model special tokens."""

        try:
            # The chunk_size contract counts body tokens, so automatic special tokens must be
            # explicitly disabled. A shared tokenizer is called concurrently, hence the lock.
            with self._lock:
                token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            raise TokenizerError("Tokenizer failed to encode text") from exc
        if not isinstance(token_ids, list) or not all(
            type(token_id) is int for token_id in token_ids
        ):
            raise TokenizerError("Tokenizer encode returned invalid token IDs")
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode body Token IDs with extra whitespace cleanup disabled."""

        try:
            # Do not let Transformers re-format whitespace; the Chunker only permits a surrounding
            # strip(). Serialized for the same reason as encode().
            with self._lock:
                text = self._tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
        except Exception as exc:
            raise TokenizerError("Tokenizer failed to decode token IDs") from exc
        if not isinstance(text, str):
            raise TokenizerError("Tokenizer decode returned non-text output")
        return text
