"""Define the vendor-neutral synchronous LLM text-generation contract and the minimal exception hierarchy."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from evidenceflow.domain.generation import LLMMessage


class LLMError(RuntimeError):
    """Common base class for every configuration and request failure of a formal LLM Client."""


class LLMConfigurationError(LLMError):
    """Signals that the Client cannot be created safely from the current non-sensitive configuration and credentials."""


class LLMRequestError(LLMError):
    """Signals that an LLM request failed or the vendor response does not satisfy the plain-text contract."""


@runtime_checkable
class LLMClient(Protocol):
    """Accept the project's message model and synchronously return the raw model text."""

    @property
    def last_usage(self) -> dict | None:
        """``prompt_tokens`` / ``completion_tokens`` of the most recent call; None when the service returned no usage."""

        ...

    def generate_json(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema_name: str,
        json_schema: dict,
    ) -> str:
        """Request JSON in the already-selected structured-output mode, return the raw text, and do not parse it at this layer."""

        ...

    def generate(self, messages: Sequence[LLMMessage]) -> str:
        """Produce one non-empty piece of text; carries no RAG or Citation business logic."""

        ...
